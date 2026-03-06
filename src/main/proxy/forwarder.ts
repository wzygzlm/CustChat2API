/**
 * Proxy Service Module - Request Forwarder
 * Forwards requests to corresponding API based on provider configuration
 */

import axios, { AxiosRequestConfig, AxiosResponse, AxiosError } from 'axios'
import http2 from 'http2'
import { PassThrough } from 'stream'
import { Account, Provider } from '../store/types'
import { ForwardResult, ChatCompletionRequest, ProxyContext, ChatCompletionTool, ToolCall } from './types'
import { proxyStatusManager } from './status'
import { storeManager } from '../store/store'
import { sessionManager } from './sessionManager'
import { DeepSeekAdapter } from './adapters/deepseek'
import { DeepSeekStreamHandler } from './adapters/deepseek-stream'
import { GLMAdapter, GLMStreamHandler } from './adapters/glm'
import { KimiAdapter, KimiStreamHandler } from './adapters/kimi'
import { QwenAdapter, QwenStreamHandler } from './adapters/qwen'
import { QwenAiAdapter, QwenAiStreamHandler } from './adapters/qwen-ai'
import { ZaiAdapter, ZaiStreamHandler } from './adapters/zai'
import { MiniMaxAdapter, MiniMaxStreamHandler } from './adapters/minimax'
import {
  isNativeFunctionCallingModel,
  buildSystemPromptWithTools,
  parseToolUse,
  formatToolResult,
  hasToolUse,
} from './promptToolUse'
import { toolsToSystemPrompt, TOOL_WRAP_HINT } from './utils/tools'
import { parseToolCallsFromText } from './utils/toolParser'

/**
 * Request Forwarder
 */
export class RequestForwarder {
  private axiosInstance = axios.create({
    timeout: 120000,
    maxBodyLength: Infinity,
    maxContentLength: Infinity,
  })

  private logInboundToolResults(
    request: ChatCompletionRequest,
    context: ProxyContext,
    provider: Provider,
    account: Account
  ): void {
    const rawTools = (request as any).tools
    const hasToolsField = Object.prototype.hasOwnProperty.call(request as any, 'tools')
    const toolsType = Array.isArray(rawTools) ? 'array' : typeof rawTools
    const toolsCount = Array.isArray(rawTools) ? rawTools.length : 0
    storeManager.addLog(
      'info',
      `[ToolFlow] Inbound tools metadata hasToolsField=${String(hasToolsField)} toolsType=${toolsType} toolsCount=${toolsCount}`,
      { requestId: context.requestId, providerId: provider.id, accountId: account.id })

    const toolMessages = (request.messages || []).filter((m: any) => m?.role === 'tool')
    if (toolMessages.length === 0) {
      return
    }

    storeManager.addLog('info', `[ToolFlow] Received tool results from client count=${toolMessages.length}`, {
      requestId: context.requestId,
      providerId: provider.id,
      accountId: account.id,
    })
  }

  private logOutboundToolResults(
    provider: Provider,
    account: Account,
    model: string,
    messages: any[]
  ): void {
    const toolMessages = (messages || []).filter((m: any) => m?.role === 'tool')
    if (toolMessages.length === 0) {
      return
    }

    storeManager.addLog('info', `[ToolFlow] Forwarding tool results to upstream count=${toolMessages.length}`, {
      providerId: provider.id,
      accountId: account.id,
    })
  }

  private logToolRouting(provider: Provider, toolResultsCount: number): void {
    if (toolResultsCount === 0) {
      return
    }

    let branch = 'generic'
    if (DeepSeekAdapter.isDeepSeekProvider(provider)) branch = 'deepseek'
    else if (GLMAdapter.isGLMProvider(provider)) branch = 'glm'
    else if (KimiAdapter.isKimiProvider(provider)) branch = 'kimi'
    else if (QwenAdapter.isQwenProvider(provider)) branch = 'qwen'
    else if (QwenAiAdapter.isQwenAiProvider(provider)) branch = 'qwen-ai'
    else if (ZaiAdapter.isZaiProvider(provider)) branch = 'zai'
    else if (MiniMaxAdapter.isMiniMaxProvider(provider)) branch = 'minimax'

    storeManager.addLog('info', `[ToolFlow] Routing tool results provider=${provider.id} branch=${branch} count=${toolResultsCount}`)
  }

  /**
   * Transform request for prompt-based tool calling
   * For models that don't support native function calling
   */
  private transformRequestForPromptToolUse(
    request: ChatCompletionRequest
  ): { messages: any[]; tools: undefined } | { messages: any[]; tools: ChatCompletionTool[] } {
    const { messages, tools, model } = request

    // If no tools or model supports native FC, return as is
    if (!tools || tools.length === 0) {
      return { messages, tools: undefined }
    }

    if (isNativeFunctionCallingModel(model)) {
      return { messages, tools }
    }

    // Transform tools to prompt format
    // Find existing system message or create one
    let systemPrompt = 'You are a helpful AI assistant.'
    const otherMessages: any[] = []

    for (const msg of messages) {
      if (msg.role === 'system') {
        systemPrompt = msg.content as string
      } else {
        otherMessages.push(msg)
      }
    }

    // Build enhanced system prompt with tools
    const enhancedSystemPrompt = buildSystemPromptWithTools(systemPrompt, tools as any[])

    // Return transformed messages with enhanced system prompt
    return {
      messages: [
        { role: 'system', content: enhancedSystemPrompt },
        ...otherMessages,
      ],
      tools: undefined,
    }
  }

  /**
   * Parse tool calls from response content
   */
  private parseToolCallsFromContent(content: string): ToolCall[] | null {
    if (!hasToolUse(content)) {
      return null
    }
    const parsed = parseToolUse(content)
    storeManager.addLog('info', `[ToolFlow] Model requested tool calls count=${parsed.length}`)
    return parsed
  }

  /**
   * Forward Chat Completions Request
   */
  async forwardChatCompletion(
    request: ChatCompletionRequest,
    account: Account,
    provider: Provider,
    actualModel: string,
    context: ProxyContext
  ): Promise<ForwardResult> {
    const startTime = Date.now()
    const config = storeManager.getConfig()
    const maxRetries = config.retryCount

    let lastError: string | undefined

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      if (attempt > 0) {
        await this.delay(5000)
      }

      try {
        const result = await this.doForward(request, account, provider, actualModel, context)

        if (result.success) {
          return result
        }

        lastError = result.error

        if (result.status && result.status < 500 && result.status !== 429) {
          break
        }
      } catch (error) {
        lastError = error instanceof Error ? error.message : 'Unknown error'
      }
    }

    return {
      success: false,
      error: lastError || 'Request failed after retries',
      latency: Date.now() - startTime,
    }
  }

  /**
   * Execute Forward
   */
  private async doForward(
    request: ChatCompletionRequest,
    account: Account,
    provider: Provider,
    actualModel: string,
    context: ProxyContext
  ): Promise<ForwardResult> {
    const startTime = Date.now()
    this.logInboundToolResults(request, context, provider, account)
    const inboundToolResultCount = (request.messages || []).filter((m: any) => m?.role === 'tool').length
    this.logToolRouting(provider, inboundToolResultCount)

    // Check if it is a DeepSeek provider, use dedicated adapter
    if (DeepSeekAdapter.isDeepSeekProvider(provider)) {
      return this.forwardDeepSeek(request, account, provider, actualModel, startTime)
    }

    // Check if it is a GLM provider, use dedicated adapter
    if (GLMAdapter.isGLMProvider(provider)) {
      return this.forwardGLM(request, account, provider, actualModel, startTime)
    }

    // Check if it is a Kimi provider, use dedicated adapter
    if (KimiAdapter.isKimiProvider(provider)) {
      return this.forwardKimi(request, account, provider, actualModel, startTime)
    }

    // Check if it is a Qwen provider, use dedicated adapter
    if (QwenAdapter.isQwenProvider(provider)) {
      return this.forwardQwen(request, account, provider, actualModel, startTime)
    }

    // Check if it is a Qwen AI (International) provider, use dedicated adapter
    if (QwenAiAdapter.isQwenAiProvider(provider)) {
      return this.forwardQwenAi(request, account, provider, actualModel, startTime)
    }

    // Check if it is a Z.ai provider, use dedicated adapter
    if (ZaiAdapter.isZaiProvider(provider)) {
      return this.forwardZai(request, account, provider, actualModel, startTime)
    }

    // Check if it is a MiniMax provider, use dedicated adapter
    if (MiniMaxAdapter.isMiniMaxProvider(provider)) {
      return this.forwardMiniMax(request, account, provider, actualModel, startTime)
    }

    try {
      const chatPath = provider.chatPath || '/chat/completions'
      const url = this.buildUrl(provider, chatPath)
      const headers = this.buildHeaders(provider, account)
      const body = this.buildRequestBody(request, actualModel, account)
      this.logOutboundToolResults(provider, account, actualModel, body.messages as any[])

      const axiosConfig: AxiosRequestConfig = {
        method: 'POST',
        url,
        headers,
        data: body,
        timeout: proxyStatusManager.getConfig().timeout,
        responseType: request.stream ? 'stream' : 'json',
        validateStatus: () => true,
      }

      const response: AxiosResponse = await this.axiosInstance.request(axiosConfig)
      const latency = Date.now() - startTime

      if (response.status >= 400) {
        return {
          success: false,
          status: response.status,
          error: this.extractErrorMessage(response),
          latency,
        }
      }

      if (request.stream) {
        return {
          success: true,
          status: response.status,
          headers: this.extractHeaders(response.headers),
          stream: response.data,
          latency,
        }
      }

      return {
        success: true,
        status: response.status,
        headers: this.extractHeaders(response.headers),
        body: response.data,
        latency,
      }
    } catch (error) {
      const latency = Date.now() - startTime

      if (error instanceof AxiosError) {
        return {
          success: false,
          status: error.response?.status,
          error: error.message,
          latency,
        }
      }

      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        latency,
      }
    }
  }

  /**
   * DeepSeek Dedicated Forward
   */
  private async forwardDeepSeek(
    request: ChatCompletionRequest,
    account: Account,
    provider: Provider,
    actualModel: string,
    startTime: number
  ): Promise<ForwardResult> {
    try {
      const transformedRequest = {
        ...request,
        messages: request.messages,
        tools: request.tools,
      }
      this.logOutboundToolResults(provider, account, actualModel, transformedRequest.messages as any[])
      const toolsCount = Array.isArray(request.tools) ? request.tools.length : 0
      const messageCount = Array.isArray(request.messages) ? request.messages.length : 0
      storeManager.addLog(
        'info',
        `[ToolFlow][DeepSeek] Outbound request snapshot stream=${!!request.stream} tools=${toolsCount} messages=${messageCount}`,
        { providerId: provider.id, accountId: account.id }
      )

      const adapter = new DeepSeekAdapter(provider, account)
      
      const sessionContext = sessionManager.getOrCreateSession({
        providerId: provider.id,
        accountId: account.id,
        model: actualModel,
      })
      
      console.log('[DeepSeek] Session context:', {
        sessionId: sessionContext.sessionId,
        providerSessionId: sessionContext.providerSessionId,
        parentMessageId: sessionContext.parentMessageId,
        isNew: sessionContext.isNew,
      })
      
      const { response, sessionId } = await adapter.chatCompletion({
        model: actualModel,
        messages: transformedRequest.messages as any,
        stream: transformedRequest.stream,
        temperature: transformedRequest.temperature,
        tools: transformedRequest.tools as any,
        tool_choice: (transformedRequest as any).tool_choice,
        sessionId: sessionContext.providerSessionId,
        parentMessageId: sessionContext.parentMessageId,
      })

      const latency = Date.now() - startTime

      if (response.status >= 400) {
        let errorMessage = `HTTP ${response.status}`
        if (response.data) {
          if (typeof response.data === 'string') {
            errorMessage = response.data
          } else if (response.data.msg) {
            errorMessage = response.data.msg
          } else if (response.data.error?.message) {
            errorMessage = response.data.error.message
          }
        }
        return {
          success: false,
          status: response.status,
          error: errorMessage,
          latency,
        }
      }

      const shouldDeleteSession = sessionManager.shouldDeleteAfterChat()
      
      const deleteSessionCallback = shouldDeleteSession
        ? async (sid: string, msgId: string) => {
            try {
              await adapter.deleteSession(sid)
              if (sessionContext.sessionId) {
                sessionManager.deleteSession(sessionContext.sessionId)
              }
            } catch (error) {
              console.error('[DeepSeek] Failed to delete session:', error)
            }
          }
        : undefined

      const handler = new DeepSeekStreamHandler(actualModel, sessionId, undefined)
      
      if (request.stream) {
        const transformedStream = await handler.handleStream(response.data)
        
        transformedStream.on('end', () => {
          const msgId = handler.getMessageId()
          if (sessionContext.sessionId && sessionManager.isMultiTurnEnabled()) {
            sessionManager.updateProviderSessionId(sessionContext.sessionId, sessionId, msgId)
          }
          if (deleteSessionCallback && shouldDeleteSession) {
            deleteSessionCallback(sessionId, msgId)
          }
        })
        
        return {
          success: true,
          status: response.status,
          headers: this.extractHeaders(response.headers),
          stream: transformedStream,
          skipTransform: true,
          latency,
        }
      }

      const result = await handler.handleNonStream(response.data)
      
      const msgId = handler.getMessageId()
      if (sessionContext.sessionId && sessionManager.isMultiTurnEnabled()) {
        sessionManager.updateProviderSessionId(sessionContext.sessionId, sessionId, msgId)
      }
      
      if (request.tools && request.tools.length > 0 && !isNativeFunctionCallingModel(request.model)) {
        const content = result?.choices?.[0]?.message?.content || ''
        const toolCalls = this.parseToolCallsFromContent(content)
        
        if (toolCalls && toolCalls.length > 0) {
          result.choices[0].message.tool_calls = toolCalls
          result.choices[0].message.content = null
          result.choices[0].finish_reason = 'tool_calls'
        }
      }
      
      if (deleteSessionCallback && shouldDeleteSession) {
        await deleteSessionCallback(sessionId, msgId)
      }

      return {
        success: true,
        status: response.status,
        headers: this.extractHeaders(response.headers),
        body: result,
        latency,
      }
    } catch (error) {
      const latency = Date.now() - startTime
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        latency,
      }
    }
  }

  /**
   * GLM Dedicated Forward
   */
  private async forwardGLM(
    request: ChatCompletionRequest,
    account: Account,
    provider: Provider,
    actualModel: string,
    startTime: number
  ): Promise<ForwardResult> {
    try {
      const transformed = this.transformRequestForPromptToolUse(request)
      const transformedRequest = {
        ...request,
        messages: transformed.messages,
        tools: transformed.tools,
      }
      this.logOutboundToolResults(provider, account, actualModel, transformedRequest.messages as any[])

      const adapter = new GLMAdapter(provider, account)
      
      const sessionContext = sessionManager.getOrCreateSession({
        providerId: provider.id,
        accountId: account.id,
        model: actualModel,
      })
      
      console.log('[GLM] Session context:', {
        sessionId: sessionContext.sessionId,
        providerSessionId: sessionContext.providerSessionId,
        isNew: sessionContext.isNew,
      })
      
      const { response, conversationId } = await adapter.chatCompletion({
        model: actualModel,
        messages: transformedRequest.messages,
        stream: transformedRequest.stream,
        temperature: transformedRequest.temperature,
        web_search: transformedRequest.web_search,
        reasoning_effort: transformedRequest.reasoning_effort,
        deep_research: transformedRequest.deep_research,
        conversationId: sessionContext.providerSessionId,
      })

      const latency = Date.now() - startTime

      if (response.status >= 400) {
        let errorMessage = `HTTP ${response.status}`
        if (response.data) {
          if (typeof response.data === 'string') {
            errorMessage = response.data
          } else if (response.data.msg) {
            errorMessage = response.data.msg
          } else if (response.data.message) {
            errorMessage = response.data.message
          } else if (response.data.error?.message) {
            errorMessage = response.data.error.message
          }
        }
        return {
          success: false,
          status: response.status,
          error: errorMessage,
          latency,
        }
      }

      const shouldDeleteSession = sessionManager.shouldDeleteAfterChat()
      const handler = new GLMStreamHandler(actualModel)
      
      if (request.stream) {
        const transformedStream = await handler.handleStream(response.data)
        
        transformedStream.on('end', () => {
          const convId = handler.getConversationId()
          if (sessionContext.sessionId && sessionManager.isMultiTurnEnabled() && convId) {
            sessionManager.updateProviderSessionId(sessionContext.sessionId, convId)
          }
          if (shouldDeleteSession && convId) {
            adapter.deleteConversation(convId).catch(err => {
              console.error('[GLM] Failed to delete session:', err)
            })
            if (sessionContext.sessionId) {
              sessionManager.deleteSession(sessionContext.sessionId)
            }
          }
        })
        
        return {
          success: true,
          status: response.status,
          headers: this.extractHeaders(response.headers),
          stream: transformedStream,
          skipTransform: true,
          latency,
        }
      }

      const result = await handler.handleNonStream(response.data)
      
      const convId = handler.getConversationId()
      console.log('[GLM] Non-stream finished, conversationId:', convId)
      if (sessionContext.sessionId && sessionManager.isMultiTurnEnabled() && convId) {
        sessionManager.updateProviderSessionId(sessionContext.sessionId, convId)
      }
      
      if (request.tools && request.tools.length > 0 && !isNativeFunctionCallingModel(request.model)) {
        const content = result?.choices?.[0]?.message?.content || ''
        const toolCalls = this.parseToolCallsFromContent(content)
        
        if (toolCalls && toolCalls.length > 0) {
          result.choices[0].message.tool_calls = toolCalls
          result.choices[0].message.content = null
          result.choices[0].finish_reason = 'tool_calls'
        }
      }
      
      if (shouldDeleteSession && convId) {
        await adapter.deleteConversation(convId)
        if (sessionContext.sessionId) {
          sessionManager.deleteSession(sessionContext.sessionId)
        }
      }

      return {
        success: true,
        status: response.status,
        headers: this.extractHeaders(response.headers),
        body: result,
        latency,
      }
    } catch (error) {
      const latency = Date.now() - startTime
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        latency,
      }
    }
  }

  private async forwardKimi(
    request: ChatCompletionRequest,
    account: Account,
    provider: Provider,
    actualModel: string,
    startTime: number
  ): Promise<ForwardResult> {
    try {
      const transformed = this.transformRequestForPromptToolUse(request)
      this.logOutboundToolResults(provider, account, actualModel, transformed.messages as any[])
      
      const adapter = new KimiAdapter(provider, account)
      
      const sessionContext = sessionManager.getOrCreateSession({
        providerId: provider.id,
        accountId: account.id,
        model: actualModel,
      })
      
      const { response, conversationId } = await adapter.chatCompletion({
        model: actualModel,
        messages: transformed.messages,
        stream: request.stream,
        temperature: request.temperature,
        enableThinking: !!request.reasoning_effort,
        enableWebSearch: !!request.web_search,
        conversationId: sessionContext.providerSessionId,
      })

      const latency = Date.now() - startTime

      if (response.status >= 400) {
        let errorMessage = `HTTP ${response.status}`
        return {
          success: false,
          status: response.status,
          error: errorMessage,
          latency,
        }
      }

      const shouldDeleteSession = sessionManager.shouldDeleteAfterChat()
      const handler = new KimiStreamHandler(actualModel, conversationId, !!request.reasoning_effort)
      
      if (request.stream) {
        const transformedStream = await handler.handleStream(response.data)
        
        transformedStream.on('end', () => {
          const convId = handler.getConversationId()
          if (sessionContext.sessionId && sessionManager.isMultiTurnEnabled() && convId) {
            sessionManager.updateProviderSessionId(sessionContext.sessionId, convId)
          }
          if (shouldDeleteSession && convId) {
            adapter.deleteConversation(convId).catch(err => {
              console.error('[Kimi] Failed to delete session:', err)
            })
            if (sessionContext.sessionId) {
              sessionManager.deleteSession(sessionContext.sessionId)
            }
          }
        })
        
        return {
          success: true,
          status: response.status,
          headers: this.extractHeaders(response.headers),
          stream: transformedStream,
          skipTransform: true,
          latency,
        }
      }

      const result = await handler.handleNonStream(response.data)
      
      const convId = handler.getConversationId()
      if (sessionContext.sessionId && sessionManager.isMultiTurnEnabled() && convId) {
        sessionManager.updateProviderSessionId(sessionContext.sessionId, convId)
      }

      if (request.tools && request.tools.length > 0 && !isNativeFunctionCallingModel(request.model)) {
        const content = result?.choices?.[0]?.message?.content || ''
        const toolCalls = this.parseToolCallsFromContent(content)
        
        if (toolCalls && toolCalls.length > 0) {
          result.choices[0].message.tool_calls = toolCalls
          result.choices[0].message.content = null
          result.choices[0].finish_reason = 'tool_calls'
        }
      }
      
      if (shouldDeleteSession && convId) {
        await adapter.deleteConversation(convId)
        if (sessionContext.sessionId) {
          sessionManager.deleteSession(sessionContext.sessionId)
        }
      }

      return {
        success: true,
        status: response.status,
        headers: this.extractHeaders(response.headers),
        body: result,
        latency,
      }
    } catch (error) {
      const latency = Date.now() - startTime
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        latency,
      }
    }
  }

  /**
   * Qwen Dedicated Forward
   */
  private async forwardQwen(
    request: ChatCompletionRequest,
    account: Account,
    provider: Provider,
    actualModel: string,
    startTime: number
  ): Promise<ForwardResult> {
    try {
      const transformed = this.transformRequestForPromptToolUse(request)
      const transformedRequest = {
        ...request,
        messages: transformed.messages,
        tools: transformed.tools,
      }
      this.logOutboundToolResults(provider, account, actualModel, transformedRequest.messages as any[])

      const adapter = new QwenAdapter(provider, account)
      
      const sessionContext = sessionManager.getOrCreateSession({
        providerId: provider.id,
        accountId: account.id,
        model: actualModel,
      })
      
      const { response, sessionId } = await adapter.chatCompletion({
        model: actualModel,
        messages: transformedRequest.messages as any,
        stream: request.stream,
        temperature: request.temperature,
        session_id: sessionContext.providerSessionId,
      })

      const latency = Date.now() - startTime

      if (response.status >= 400) {
        let errorMessage = `HTTP ${response.status}`
        return {
          success: false,
          status: response.status,
          error: errorMessage,
          latency,
        }
      }

      const shouldDeleteSession = sessionManager.shouldDeleteAfterChat()
      const deleteSessionCallback = shouldDeleteSession
        ? async (sid: string) => {
            try {
              await adapter.deleteSession(sid)
              if (sessionContext.sessionId) {
                sessionManager.deleteSession(sessionContext.sessionId)
              }
            } catch (err) {
              console.error('[Qwen] Failed to delete session:', err)
            }
          }
        : undefined

      const handler = new QwenStreamHandler(actualModel, deleteSessionCallback)

      if (request.stream) {
        const transformedStream = await handler.handleStream(response.data, response)
        
        transformedStream.on('end', () => {
          const sid = handler.getSessionId()
          if (sessionContext.sessionId && sessionManager.isMultiTurnEnabled() && sid) {
            sessionManager.updateProviderSessionId(sessionContext.sessionId, sid)
          }
        })
        
        return {
          success: true,
          status: response.status,
          headers: this.extractHeaders(response.headers),
          stream: transformedStream,
          skipTransform: true,
          latency,
        }
      }

      const result = await handler.handleNonStream(response.data, response)
      
      const sid = handler.getSessionId()
      if (sessionContext.sessionId && sessionManager.isMultiTurnEnabled() && sid) {
        sessionManager.updateProviderSessionId(sessionContext.sessionId, sid)
      }

      if (request.tools && request.tools.length > 0 && !isNativeFunctionCallingModel(request.model)) {
        const content = result?.choices?.[0]?.message?.content || ''
        const toolCalls = this.parseToolCallsFromContent(content)
        
        if (toolCalls && toolCalls.length > 0) {
          result.choices[0].message.tool_calls = toolCalls
          result.choices[0].message.content = null
          result.choices[0].finish_reason = 'tool_calls'
        }
      }

      if (deleteSessionCallback && sid) {
        await deleteSessionCallback(sid)
      }

      return {
        success: true,
        status: response.status,
        headers: this.extractHeaders(response.headers),
        body: result,
        latency,
      }
    } catch (error) {
      const latency = Date.now() - startTime
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        latency,
      }
    }
  }

  /**
   * Qwen AI (International) Dedicated Forward
   */
  private async forwardQwenAi(
    request: ChatCompletionRequest,
    account: Account,
    provider: Provider,
    actualModel: string,
    startTime: number
  ): Promise<ForwardResult> {
    try {
      const transformed = this.transformRequestForPromptToolUse(request)
      this.logOutboundToolResults(provider, account, actualModel, transformed.messages as any[])
      
      const adapter = new QwenAiAdapter(provider, account)
      
      const sessionContext = sessionManager.getOrCreateSession({
        providerId: provider.id,
        accountId: account.id,
        model: actualModel,
      })
      
      const { response, chatId } = await adapter.chatCompletion({
        model: actualModel,
        messages: transformed.messages as any,
        stream: request.stream,
        temperature: request.temperature,
        enable_thinking: !!request.reasoning_effort,
        chatId: sessionContext.providerSessionId,
      })

      const latency = Date.now() - startTime

      if (response.status >= 400) {
        let errorMessage = `HTTP ${response.status}`
        return {
          success: false,
          status: response.status,
          error: errorMessage,
          latency,
        }
      }

      const shouldDeleteSession = sessionManager.shouldDeleteAfterChat()
      const deleteChatCallback = shouldDeleteSession
        ? async (cid: string) => {
            try {
              await adapter.deleteChat(cid)
              if (sessionContext.sessionId) {
                sessionManager.deleteSession(sessionContext.sessionId)
              }
            } catch (err) {
              console.error('[QwenAI] Failed to delete chat:', err)
            }
          }
        : undefined

      const handler = new QwenAiStreamHandler(actualModel, deleteChatCallback)
      handler.setChatId(chatId)

      if (request.stream) {
        const transformedStream = await handler.handleStream(response.data)
        
        transformedStream.on('end', () => {
          const msgId = handler.getMessageId()
          if (sessionContext.sessionId && sessionManager.isMultiTurnEnabled() && chatId) {
            sessionManager.updateProviderSessionId(sessionContext.sessionId, chatId, msgId)
          }
        })

        return {
          success: true,
          status: response.status,
          headers: this.extractHeaders(response.headers),
          stream: transformedStream,
          skipTransform: true,
          latency,
        }
      }

      const result = await handler.handleNonStream(response.data)
      
      if (sessionContext.sessionId && sessionManager.isMultiTurnEnabled() && chatId) {
        sessionManager.updateProviderSessionId(sessionContext.sessionId, chatId)
      }

      if (request.tools && request.tools.length > 0 && !isNativeFunctionCallingModel(request.model)) {
        const content = result?.choices?.[0]?.message?.content || ''
        const toolCalls = this.parseToolCallsFromContent(content)
        
        if (toolCalls && toolCalls.length > 0) {
          result.choices[0].message.tool_calls = toolCalls
          result.choices[0].message.content = null
          result.choices[0].finish_reason = 'tool_calls'
        }
      }

      if (deleteChatCallback) {
        await deleteChatCallback(chatId)
      }

      return {
        success: true,
        status: response.status,
        headers: this.extractHeaders(response.headers),
        body: result,
        latency,
      }
    } catch (error) {
      const latency = Date.now() - startTime
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        latency,
      }
    }
  }

  /**
   * Z.ai Dedicated Forward
   */
  private async forwardZai(
    request: ChatCompletionRequest,
    account: Account,
    provider: Provider,
    actualModel: string,
    startTime: number
  ): Promise<ForwardResult> {
    console.log('[forwardZai] actualModel:', actualModel)
    console.log('[forwardZai] provider.modelMappings:', provider.modelMappings)
    try {
      const transformed = this.transformRequestForPromptToolUse(request)
      this.logOutboundToolResults(provider, account, actualModel, transformed.messages as any[])
      
      const adapter = new ZaiAdapter(provider, account)
      
      const sessionContext = sessionManager.getOrCreateSession({
        providerId: provider.id,
        accountId: account.id,
        model: actualModel,
      })
      
      console.log('[Z.ai] Session context:', {
        sessionId: sessionContext.sessionId,
        providerSessionId: sessionContext.providerSessionId,
        isNew: sessionContext.isNew,
      })
      
      const { response, chatId, userMessageId } = await adapter.chatCompletion({
        model: actualModel,
        messages: transformed.messages as any,
        stream: request.stream,
        temperature: request.temperature,
        web_search: request.web_search,
        reasoning_effort: request.reasoning_effort,
        chatId: sessionContext.providerSessionId,
        parentMessageId: sessionContext.parentMessageId,
      })

      const latency = Date.now() - startTime

      if (response.status >= 400) {
        let errorMessage = `HTTP ${response.status}`
        return {
          success: false,
          status: response.status,
          error: errorMessage,
          latency,
        }
      }

      const shouldDeleteSession = sessionManager.shouldDeleteAfterChat()
      const deleteChatCallback = shouldDeleteSession
        ? async (cid: string) => {
            try {
              await adapter.deleteChat(cid)
              if (sessionContext.sessionId) {
                sessionManager.deleteSession(sessionContext.sessionId)
              }
            } catch (error) {
              console.error('[Z.ai] Failed to delete chat:', error)
            }
          }
        : undefined

      const hasToolsInRequest = Array.isArray((request as any).tools) && (request as any).tools.length > 0
      const handler = new ZaiStreamHandler(actualModel, deleteChatCallback, hasToolsInRequest)
      handler.setChatId(chatId)
      
      if (request.stream !== false) {
        const transformedStream = await handler.handleStream(response.data)
        
        transformedStream.on('end', async () => {
          // For Z.ai, we need to call the batch API to get the assistant message ID
          if (sessionContext.sessionId && sessionManager.isMultiTurnEnabled() && chatId && userMessageId) {
            try {
              const assistantMsgId = await adapter.getLatestMessageId(chatId, userMessageId)
              console.log('[Z.ai] Got assistant message ID from batch API:', assistantMsgId || 'N/A')
              if (assistantMsgId) {
                sessionManager.updateProviderSessionId(sessionContext.sessionId, chatId, assistantMsgId)
              }
            } catch (error) {
              console.error('[Z.ai] Failed to get assistant message ID:', error)
            }
          }
        })
        
        return {
          success: true,
          status: response.status,
          headers: this.extractHeaders(response.headers),
          stream: transformedStream,
          skipTransform: true,
          latency,
        }
      }

      const result = await handler.handleNonStream(response)
      
      console.log('[Z.ai] Non-stream finished, chatId:', chatId)
      
      // For Z.ai, we need to call the batch API to get the assistant message ID
      let assistantMsgId: string | undefined
      if (sessionContext.sessionId && sessionManager.isMultiTurnEnabled() && chatId && userMessageId) {
        try {
          const msgId = await adapter.getLatestMessageId(chatId, userMessageId)
          assistantMsgId = msgId || undefined
          console.log('[Z.ai] Got assistant message ID from batch API:', assistantMsgId || 'N/A')
        } catch (error) {
          console.error('[Z.ai] Failed to get assistant message ID:', error)
        }
      }
      
      if (sessionContext.sessionId && sessionManager.isMultiTurnEnabled() && chatId) {
        sessionManager.updateProviderSessionId(sessionContext.sessionId, chatId, assistantMsgId)
      }

      if (hasToolsInRequest && !isNativeFunctionCallingModel(request.model)) {
        const content = result?.choices?.[0]?.message?.content || ''
        const toolCalls = this.parseToolCallsFromContent(content)
        
        if (toolCalls && toolCalls.length > 0) {
          result.choices[0].message.tool_calls = toolCalls
          result.choices[0].message.content = null
          result.choices[0].finish_reason = 'tool_calls'
        }
      }
      
      if (deleteChatCallback) {
        await deleteChatCallback(chatId)
      }

      return {
        success: true,
        status: response.status,
        headers: this.extractHeaders(response.headers),
        body: result,
        latency,
      }
    } catch (error) {
      const latency = Date.now() - startTime
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        latency,
      }
    }
  }

  /**
   * MiniMax Dedicated Forward
   */
  private async forwardMiniMax(
    request: ChatCompletionRequest,
    account: Account,
    provider: Provider,
    actualModel: string,
    startTime: number
  ): Promise<ForwardResult> {
    console.log('[forwardMiniMax] actualModel:', actualModel)
    console.log('[forwardMiniMax] provider.modelMappings:', provider.modelMappings)
    try {
      const transformed = this.transformRequestForPromptToolUse(request)
      this.logOutboundToolResults(provider, account, actualModel, transformed.messages as any[])
      
      const adapter = new MiniMaxAdapter(provider, account)
      
      const sessionContext = sessionManager.getOrCreateSession({
        providerId: provider.id,
        accountId: account.id,
        model: actualModel,
      })
      
      const { response, stream, chatId } = await adapter.chatCompletion({
        model: actualModel,
        messages: transformed.messages as any,
        stream: request.stream,
        temperature: request.temperature,
        chatId: sessionContext.providerSessionId,
      })

      const latency = Date.now() - startTime

      if (response && response.status >= 400) {
        let errorMessage = `HTTP ${response.status}`
        return {
          success: false,
          status: response.status,
          error: errorMessage,
          latency,
        }
      }

      const shouldDeleteSession = sessionManager.shouldDeleteAfterChat()
      const deleteChatCallback = shouldDeleteSession
        ? async (cid: string) => {
            try {
              await adapter.deleteChat(cid)
              if (sessionContext.sessionId) {
                sessionManager.deleteSession(sessionContext.sessionId)
              }
            } catch (error) {
              console.error('[MiniMax] Failed to delete chat:', error)
            }
          }
        : undefined

      if (request.stream !== false && stream) {
        console.log('[forwardMiniMax] Using polling stream')
        
        stream.stream.on('end', () => {
          if (sessionContext.sessionId && sessionManager.isMultiTurnEnabled() && chatId) {
            sessionManager.updateProviderSessionId(sessionContext.sessionId, chatId)
          }
          if (deleteChatCallback && shouldDeleteSession) {
            deleteChatCallback(chatId)
          }
        })
        
        return {
          success: true,
          status: 200,
          headers: {},
          stream: stream.stream as any,
          skipTransform: true,
          latency,
        }
      }

      if (response) {
        if (sessionContext.sessionId && sessionManager.isMultiTurnEnabled() && chatId) {
          sessionManager.updateProviderSessionId(sessionContext.sessionId, chatId)
        }
        
        if (request.tools && request.tools.length > 0 && !isNativeFunctionCallingModel(request.model)) {
          const content = response.data?.choices?.[0]?.message?.content || ''
          const toolCalls = this.parseToolCallsFromContent(content)
          
          if (toolCalls && toolCalls.length > 0) {
            response.data.choices[0].message.tool_calls = toolCalls
            response.data.choices[0].message.content = null
            response.data.choices[0].finish_reason = 'tool_calls'
          }
        }
        
        if (deleteChatCallback && shouldDeleteSession) {
          await deleteChatCallback(chatId)
        }

        return {
          success: true,
          status: response.status,
          headers: this.extractHeaders(response.headers),
          body: response.data,
          latency,
        }
      }

      return {
        success: false,
        error: 'No response or stream received',
        latency,
      }
    } catch (error) {
      const latency = Date.now() - startTime
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        latency,
      }
    }
  }

  /**
   * Build URL
   */
  private buildUrl(provider: Provider, path: string): string {
    let baseUrl = provider.apiEndpoint

    if (baseUrl.endsWith('/')) {
      baseUrl = baseUrl.slice(0, -1)
    }

    if (!path.startsWith('/')) {
      path = '/' + path
    }

    if (baseUrl.includes('/v1') && path.startsWith('/v1')) {
      path = path.slice(3)
    }

    return `${baseUrl}${path}`
  }

  /**
   * Build Request Headers
   */
  private buildHeaders(provider: Provider, account: Account): Record<string, string> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...provider.headers,
    }

    const credentials = account.credentials

    if (credentials.token) {
      headers['Authorization'] = `Bearer ${credentials.token}`
    } else if (credentials.apiKey) {
      headers['Authorization'] = `Bearer ${credentials.apiKey}`
    } else if (credentials.accessToken) {
      headers['Authorization'] = `Bearer ${credentials.accessToken}`
    } else if (credentials.refreshToken) {
      headers['Authorization'] = `Bearer ${credentials.refreshToken}`
    }

    if (credentials.cookie) {
      headers['Cookie'] = credentials.cookie
    }

    if (credentials.sessionKey) {
      headers['X-Session-Key'] = credentials.sessionKey
    }

    return headers
  }

  /**
   * Build Request Body
   */
  private buildRequestBody(
    request: ChatCompletionRequest,
    actualModel: string,
    account: Account
  ): any {
    const body: any = {
      model: actualModel,
      messages: request.messages,
      stream: request.stream || false,
    }

    if (request.temperature !== undefined) {
      body.temperature = request.temperature
    }

    if (request.top_p !== undefined) {
      body.top_p = request.top_p
    }

    if (request.n !== undefined) {
      body.n = request.n
    }

    if (request.stop !== undefined) {
      body.stop = request.stop
    }

    if (request.max_tokens !== undefined) {
      body.max_tokens = request.max_tokens
    }

    if (request.presence_penalty !== undefined) {
      body.presence_penalty = request.presence_penalty
    }

    if (request.frequency_penalty !== undefined) {
      body.frequency_penalty = request.frequency_penalty
    }

    if (request.logit_bias !== undefined) {
      body.logit_bias = request.logit_bias
    }

    if (request.user !== undefined) {
      body.user = request.user
    }

    return body
  }

  /**
   * Extract Response Headers
   */
  private extractHeaders(headers: any): Record<string, string> {
    const result: Record<string, string> = {}

    for (const [key, value] of Object.entries(headers)) {
      if (typeof value === 'string') {
        result[key] = value
      } else if (Array.isArray(value)) {
        result[key] = value.join(', ')
      }
    }

    return result
  }

  /**
   * Extract Error Message
   */
  private extractErrorMessage(response: AxiosResponse): string {
    if (response.data) {
      if (typeof response.data === 'string') {
        return response.data
      }

      if (response.data.error?.message) {
        return response.data.error.message
      }

      if (response.data.message) {
        return response.data.message
      }

      if (response.data.msg) {
        return response.data.msg
      }

      try {
        return JSON.stringify(response.data)
      } catch {
        return 'Unknown error'
      }
    }

    return `HTTP ${response.status}`
  }

  /**
   * Delay
   */
  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms))
  }

  /**
   * Forward Request to Specified URL
   */
  async forwardToUrl(
    url: string,
    method: string,
    headers: Record<string, string>,
    body: any,
    isStream: boolean = false
  ): Promise<ForwardResult> {
    const startTime = Date.now()

    try {
      const config: AxiosRequestConfig = {
        method,
        url,
        headers,
        data: body,
        timeout: proxyStatusManager.getConfig().timeout,
        responseType: isStream ? 'stream' : 'json',
        validateStatus: () => true,
      }

      const response: AxiosResponse = await this.axiosInstance.request(config)
      const latency = Date.now() - startTime

      if (response.status >= 400) {
        return {
          success: false,
          status: response.status,
          error: this.extractErrorMessage(response),
          latency,
        }
      }

      if (isStream) {
        return {
          success: true,
          status: response.status,
          headers: this.extractHeaders(response.headers),
          stream: response.data,
          latency,
        }
      }

      return {
        success: true,
        status: response.status,
        headers: this.extractHeaders(response.headers),
        body: response.data,
        latency,
      }
    } catch (error) {
      const latency = Date.now() - startTime

      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        latency,
      }
    }
  }
}

export const requestForwarder = new RequestForwarder()
export default requestForwarder
