import { storeManager } from '../store/store'
import { ProxyServer } from '../proxy/server'
import { proxyStatusManager } from '../proxy/status'
import type { ProxyStatus } from '../../shared/types'
import type { ProxyStatistics } from '../proxy/types'

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

let proxyServer: ProxyServer | null = null
let proxyStartTime: number | null = null

function getStatus(): ProxyStatus {
  const isRunning = proxyServer !== null
  return {
    isRunning,
    port: proxyStatusManager.getPort(),
    uptime: proxyStartTime && isRunning ? Date.now() - proxyStartTime : 0,
    connections: proxyStatusManager.getStatistics().activeConnections,
  }
}

function sendMessage(message: ServiceResponseMessage | ServiceEventMessage): void {
  if (typeof process.send === 'function') {
    process.send(message)
  }
}

function emitStatusChanged(): void {
  sendMessage({
    kind: 'event',
    type: 'status-changed',
    payload: getStatus(),
  })
}

async function startProxy(port?: number): Promise<ProxyStatus> {
  if (proxyServer) {
    return getStatus()
  }

  const config = storeManager.getConfig()
  const proxyPort = typeof port === 'number' && port > 0 ? port : config.proxyPort

  proxyServer = new ProxyServer()
  const success = await proxyServer.start(proxyPort)
  if (!success) {
    proxyServer = null
    throw new Error(`Failed to start proxy service on port ${proxyPort}`)
  }

  proxyStartTime = Date.now()
  emitStatusChanged()
  return getStatus()
}

async function stopProxy(): Promise<ProxyStatus> {
  if (!proxyServer) {
    proxyStartTime = null
    emitStatusChanged()
    return getStatus()
  }

  const server = proxyServer
  proxyServer = null

  try {
    await server.stop()
  } finally {
    proxyStartTime = null
    emitStatusChanged()
  }

  return getStatus()
}

function getStatistics(): ProxyStatistics {
  return proxyStatusManager.getStatistics()
}

async function handleRequest(message: ServiceRequestMessage): Promise<void> {
  try {
    let result: unknown

    switch (message.type) {
      case 'start':
        result = await startProxy(typeof message.payload?.port === 'number' ? message.payload.port : undefined)
        break
      case 'stop':
        result = await stopProxy()
        break
      case 'get-status':
        result = getStatus()
        break
      case 'get-statistics':
        result = getStatistics()
        break
      case 'reset-statistics':
        proxyStatusManager.resetStatistics()
        result = undefined
        break
      default:
        throw new Error(`Unsupported service request: ${String((message as { type?: string }).type)}`)
    }

    sendMessage({
      kind: 'response',
      id: message.id,
      ok: true,
      result,
    })
  } catch (error) {
    sendMessage({
      kind: 'response',
      id: message.id,
      ok: false,
      error: error instanceof Error ? error.message : 'Unknown proxy service error',
    })
  }
}

async function bootstrap(): Promise<void> {
  await storeManager.initialize()

  process.on('message', (message: ServiceRequestMessage) => {
    if (!message || message.kind !== 'request') {
      return
    }
    void handleRequest(message)
  })

  process.on('disconnect', () => {
    void stopProxy().finally(() => process.exit(0))
  })

  process.on('SIGTERM', () => {
    void stopProxy().finally(() => process.exit(0))
  })

  process.on('SIGINT', () => {
    void stopProxy().finally(() => process.exit(0))
  })

  emitStatusChanged()
}

void bootstrap().catch((error) => {
  console.error('[ProxyService] Bootstrap failed:', error)
  process.exit(1)
})
