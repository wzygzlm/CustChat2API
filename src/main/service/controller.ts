import { fork, type ChildProcess } from 'child_process'
import { join } from 'path'
import type { ProxyStatistics } from '../proxy/types'
import type { ProxyStatus } from '../../shared/types'
import { ProxyServer } from '../proxy/server'
import { proxyStatusManager } from '../proxy/status'

type ServiceRequestType = 'start' | 'stop' | 'get-status' | 'get-statistics' | 'reset-statistics'

interface ServiceRequestMessage {
  kind: 'request'
  id: number
  type: ServiceRequestType
  payload?: Record<string, unknown>
}

interface ServiceResponseMessage {
  kind: 'response'
  id: number
  ok: boolean
  result?: unknown
  error?: string
}

interface ServiceEventMessage {
  kind: 'event'
  type: 'status-changed'
  payload: ProxyStatus
}

type ServiceMessage = ServiceResponseMessage | ServiceEventMessage

function createStoppedStatus(): ProxyStatus {
  return {
    isRunning: false,
    port: 0,
    uptime: 0,
    connections: 0,
  }
}

function createEmptyStatistics(): ProxyStatistics {
  return {
    totalRequests: 0,
    successRequests: 0,
    failedRequests: 0,
    avgLatency: 0,
    requestsPerMinute: 0,
    activeConnections: 0,
    modelUsage: {},
    providerUsage: {},
    accountUsage: {},
  }
}

export class ProxyServiceController {
  private child: ChildProcess | null = null
  private localProxyServer: ProxyServer | null = null
  private localProxyStartTime: number | null = null
  private nextRequestId = 1
  private pending = new Map<number, { resolve: (value: unknown) => void; reject: (reason?: unknown) => void }>()
  private cachedStatus: ProxyStatus = createStoppedStatus()
  private onStatusChanged?: (status: ProxyStatus) => void
  private readyPromise: Promise<void> | null = null
  private resolveReady: (() => void) | null = null
  private rejectReady: ((reason?: unknown) => void) | null = null

  setStatusListener(listener?: (status: ProxyStatus) => void): void {
    this.onStatusChanged = listener
  }

  getCachedStatus(): ProxyStatus {
    return { ...this.cachedStatus }
  }

  async start(port?: number): Promise<boolean> {
    if (this.localProxyServer) {
      return true
    }

    try {
      this.ensureChildProcess()
      await this.waitUntilReady()
      const status = await this.sendRequest<ProxyStatus>('start', { port })
      this.updateCachedStatus(status)
      return status.isRunning
    } catch (error) {
      console.warn('[ProxyService] Falling back to in-process proxy runtime:', error)
      const status = await this.startLocalProxy(port)
      this.updateCachedStatus(status)
      return status.isRunning
    }
  }

  async stop(): Promise<boolean> {
    if (this.localProxyServer) {
      const status = await this.stopLocalProxy()
      this.updateCachedStatus(status)
      return true
    }

    if (!this.child) {
      this.updateCachedStatus(createStoppedStatus())
      return true
    }

    try {
      const status = await this.sendRequest<ProxyStatus>('stop')
      this.updateCachedStatus(status)
    } finally {
      this.teardownChild()
    }

    return true
  }

  async getStatus(): Promise<ProxyStatus> {
    if (this.localProxyServer) {
      const status = this.getLocalProxyStatus()
      this.updateCachedStatus(status)
      return status
    }

    if (!this.child) {
      return this.getCachedStatus()
    }

    await this.waitUntilReady()
    const status = await this.sendRequest<ProxyStatus>('get-status')
    this.updateCachedStatus(status)
    return status
  }

  async getStatistics(): Promise<ProxyStatistics> {
    if (this.localProxyServer) {
      return proxyStatusManager.getStatistics()
    }

    if (!this.child) {
      return createEmptyStatistics()
    }
    await this.waitUntilReady()
    return this.sendRequest<ProxyStatistics>('get-statistics')
  }

  async resetStatistics(): Promise<void> {
    if (this.localProxyServer) {
      proxyStatusManager.resetStatistics()
      return
    }

    if (!this.child) {
      return
    }
    await this.waitUntilReady()
    await this.sendRequest('reset-statistics')
  }

  async shutdown(): Promise<void> {
    if (!this.child) {
      return
    }

    try {
      await this.stop()
    } catch {
      this.teardownChild()
    }
  }

  private ensureChildProcess(): void {
    if (this.child && this.child.connected) {
      return
    }

    const serviceEntry = join(__dirname, 'service.js')
    this.readyPromise = new Promise<void>((resolve, reject) => {
      this.resolveReady = resolve
      this.rejectReady = reject
    })
    try {
      this.child = fork(serviceEntry, [], {
        env: {
          ...process.env,
          ELECTRON_RUN_AS_NODE: '1',
          CHAT2API_SERVICE_PROCESS: '1',
        },
        stdio: ['ignore', 'ignore', 'pipe', 'ipc'],
      })
    } catch (error) {
      this.child = null
      this.readyPromise = null
      this.resolveReady = null
      this.rejectReady = null
      throw error
    }

    this.child.stderr?.on('data', (chunk: Buffer | string) => {
      const text = chunk.toString().trim()
      if (text) {
        console.error(`[ProxyService] ${text}`)
      }
    })

    this.child.on('message', (message: ServiceMessage) => {
      this.handleMessage(message)
    })

    this.child.once('exit', () => {
      this.rejectPendingRequests(new Error('Proxy service process exited'))
      this.rejectReady?.(new Error('Proxy service process exited before becoming ready'))
      this.child = null
      this.readyPromise = null
      this.resolveReady = null
      this.rejectReady = null
      this.updateCachedStatus(createStoppedStatus())
    })

    this.child.once('error', (error) => {
      this.rejectPendingRequests(error)
      this.rejectReady?.(error)
      this.child = null
      this.readyPromise = null
      this.resolveReady = null
      this.rejectReady = null
      this.updateCachedStatus(createStoppedStatus())
    })
  }

  private async sendRequest<T = unknown>(
    type: ServiceRequestType,
    payload?: Record<string, unknown>
  ): Promise<T> {
    this.ensureChildProcess()

    const child = this.child
    if (!child?.connected) {
      throw new Error('Proxy service process is not available')
    }

    const id = this.nextRequestId++
    const message: ServiceRequestMessage = {
      kind: 'request',
      id,
      type,
      payload,
    }

    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve, reject })
      child.send(message)
    })
  }

  private handleMessage(message: ServiceMessage): void {
    if (!message || typeof message !== 'object') {
      return
    }

    if (message.kind === 'event' && message.type === 'status-changed') {
      this.resolveReady?.()
      this.resolveReady = null
      this.rejectReady = null
      this.updateCachedStatus(message.payload)
      return
    }

    if (message.kind !== 'response') {
      return
    }

    const pending = this.pending.get(message.id)
    if (!pending) {
      return
    }

    this.pending.delete(message.id)

    if (message.ok) {
      pending.resolve(message.result)
    } else {
      pending.reject(new Error(message.error || 'Unknown proxy service error'))
    }
  }

  private updateCachedStatus(status: ProxyStatus): void {
    this.cachedStatus = { ...status }
    this.onStatusChanged?.(this.getCachedStatus())
  }

  private async waitUntilReady(): Promise<void> {
    if (!this.readyPromise) {
      return
    }
    await this.readyPromise
  }

  private rejectPendingRequests(error: Error): void {
    for (const { reject } of this.pending.values()) {
      reject(error)
    }
    this.pending.clear()
  }

  private teardownChild(): void {
    if (!this.child) {
      return
    }

    this.child.removeAllListeners()
    if (this.child.connected) {
      this.child.disconnect()
    }
    if (!this.child.killed) {
      this.child.kill()
    }
    this.child = null
    this.readyPromise = null
    this.resolveReady = null
    this.rejectReady = null
  }

  private getLocalProxyStatus(): ProxyStatus {
    return {
      isRunning: this.localProxyServer !== null,
      port: proxyStatusManager.getPort(),
      uptime:
        this.localProxyStartTime && this.localProxyServer
          ? Date.now() - this.localProxyStartTime
          : 0,
      connections: proxyStatusManager.getStatistics().activeConnections,
    }
  }

  private async startLocalProxy(port?: number): Promise<ProxyStatus> {
    if (this.localProxyServer) {
      return this.getLocalProxyStatus()
    }

    const server = new ProxyServer()
    const success = await server.start(port)
    if (!success) {
      throw new Error(`Failed to start in-process proxy on port ${port ?? 0}`)
    }

    this.localProxyServer = server
    this.localProxyStartTime = Date.now()
    return this.getLocalProxyStatus()
  }

  private async stopLocalProxy(): Promise<ProxyStatus> {
    if (!this.localProxyServer) {
      this.localProxyStartTime = null
      return createStoppedStatus()
    }

    const server = this.localProxyServer
    this.localProxyServer = null

    try {
      await server.stop()
    } finally {
      this.localProxyStartTime = null
    }

    return this.getLocalProxyStatus()
  }
}

export const proxyServiceController = new ProxyServiceController()
