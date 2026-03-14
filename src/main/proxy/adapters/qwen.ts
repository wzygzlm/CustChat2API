/**
 * Qwen Adapter
 * Implements Qwen (Tongyi Qianwen) web API protocol
 * Based on new chat2.qianwen.com API
 */

import axios, { AxiosResponse } from 'axios'
import { PassThrough } from 'stream'
import { createGunzip, createInflate, createBrotliDecompress } from 'zlib'
import * as ZstdCodec from 'zstd-codec'
import { createParser } from 'eventsource-parser'
import { Account, Provider } from '../../store/types'
import { hasToolUse, parseToolUse, ToolCall } from '../promptToolUse'

const QWEN_API_BASE = 'https://chat2.qianwen.com'

const MODEL_MAP: Record<string, string> = {
  'Qwen3': 'tongyi-qwen3-max-model-agent',
  'Qwen3-Max': 'tongyi-qwen3-max-model-agent',
  'Qwen3-Max-Thinking': 'tongyi-qwen3-max-thinking-agent',
  'Qwen3-Plus': 'tongyi-qwen-plus-agent',
  'Qwen3.5-Plus': 'Qwen3.5-Plus',
  'Qwen3-Flash': 'qwen3-flash',
  'Qwen3-Coder': 'qwen3-coder-plus',
}

const DEFAULT_HEADERS = {
  Accept: 'application/json, text/event-stream, text/plain, */*',
  'Accept-Language': 'zh-CN,zh;q=0.9',
  'Cache-Control': 'no-cache',
  Origin: 'https://www.qianwen.com',
  Pragma: 'no-cache',
  'Sec-Ch-Ua': '"Chromium";v="145", "Not(A:Brand";v="24", "Google Chrome";v="145"',
  'Sec-Ch-Ua-Mobile': '?0',
  'Sec-Ch-Ua-Platform': '"macOS"',
  'Sec-Fetch-Dest': 'empty',
  'Sec-Fetch-Mode': 'cors',
  'Sec-Fetch-Site': 'same-site',
  Referer: 'https://www.qianwen.com/',
  'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36',
}

interface QwenMessage {
  role: 'user' | 'assistant' | 'system' | 'tool'
  content: string | any[] | null
  tool_call_id?: string
  tool_calls?: Array<{
    id?: string
    function?: {
      name?: string
      arguments?: string
    }
  }>
}

interface ChatCompletionRequest {
  model: string
  messages: QwenMessage[]
  stream?: boolean
  temperature?: number
  session_id?: string
}

function uuid(separator: boolean = true): string {
  const id = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0
    const v = c === 'x' ? r : (r & 0x3) | 0x8
    return v.toString(16)
  })
  return separator ? id : id.replace(/-/g, '')
}

function generateNonce(): string {
  const chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
  let result = ''
  for (let i = 0; i < 12; i++) {
    result += chars.charAt(Math.floor(Math.random() * chars.length))
  }
  return result
}

function extractTextContent(content: string | any[]): string {
  if (typeof content === 'string') {
    return content
  }
  if (Array.isArray(content)) {
    return content
      .filter((item) => item.type === 'text')
      .map((item) => item.text || '')
      .join('\n')
  }
  return ''
}

export class QwenAdapter {
  private provider: Provider
  private account: Account
  private axiosInstance = axios.create({
    timeout: 120000,
    maxBodyLength: Infinity,
    maxContentLength: Infinity,
  })

  constructor(provider: Provider, account: Account) {
    this.provider = provider
    this.account = account
  }

  private getTicket(): string {
    const credentials = this.account.credentials
    return credentials.ticket || credentials.tongyi_sso_ticket || ''
  }

  private mapModel(model: string): string {
    if (MODEL_MAP[model]) {
      return MODEL_MAP[model]
    }
    return model
  }

  private stringifyContent(content: QwenMessage['content']): string {
    if (content == null) return ''
    if (typeof content === 'string') return content
    if (Array.isArray(content)) {
      return content
        .map((part: any) => {
          if (typeof part === 'string') return part
          if (!part || typeof part !== 'object') return String(part ?? '')
          if (part.type === 'text' && typeof part.text === 'string') return part.text
          return JSON.stringify(part)
        })
        .filter(Boolean)
        .join('\n')
    }
    if (typeof content === 'object') return JSON.stringify(content)
    return String(content)
  }

  private normalizeMessages(messages: QwenMessage[]): QwenMessage[] {
    return messages.map((msg) => {
      if (msg.role === 'assistant' && Array.isArray(msg.tool_calls) && msg.tool_calls.length > 0) {
        const toolCallsText = msg.tool_calls
          .map((tc) => `[call:${tc?.function?.name || 'tool'}]${tc?.function?.arguments || '{}'}[/call]`)
          .join('\n')
        return { ...msg, content: `[function_calls]\n${toolCallsText}\n[/function_calls]`, tool_calls: undefined }
      }

      if (msg.role === 'tool') {
        const normalizedToolContent = this.stringifyContent(msg.content)
        return {
          role: 'user',
          content: `[TOOL_RESULT for ${msg.tool_call_id || 'tool_call'}] ${normalizedToolContent}`,
        }
      }

      return {
        ...msg,
        content: this.stringifyContent(msg.content),
      }
    })
  }

  async chatCompletion(request: ChatCompletionRequest): Promise<{
    response: AxiosResponse
    sessionId: string
    reqId: string
  }> {
    const ticket = this.getTicket()
    if (!ticket) {
      throw new Error('Qwen ticket not configured, please add ticket in account settings')
    }

    const reqId = uuid(false)
    const sessionId = request.session_id || uuid(false)
    const actualModel = this.mapModel(request.model)
    
    const normalizedMessages = this.normalizeMessages(request.messages)

    // Find system message and user message
    let systemPrompt = ''
    let userContent = ''
    
    for (const msg of normalizedMessages) {
      if (msg.role === 'system') {
        systemPrompt = this.stringifyContent(msg.content)
      } else if (msg.role === 'user') {
        userContent = this.stringifyContent(msg.content)
      }
    }

    const hasToolContext = normalizedMessages.some(
      (msg) =>
        msg.role === 'tool' ||
        (msg.role === 'assistant' && Array.isArray((msg as any).tool_calls) && (msg as any).tool_calls.length > 0) ||
        (typeof msg.content === 'string' && (msg.content.includes('[function_calls]') || msg.content.includes('[TOOL_RESULT for ')))
    )

    // Keep previous concise behavior for normal chat; switch to full conversation only when tool context is present.
    const finalContent = hasToolContext
      ? normalizedMessages
          .map((msg) => `${msg.role === 'system' ? 'system' : msg.role === 'assistant' ? 'assistant' : 'user'}: ${this.stringifyContent(msg.content)}`)
          .join('\n\n')
      : systemPrompt
        ? `${systemPrompt}\n\nUser: ${userContent}`
        : userContent

    const timestamp = Date.now()
    const nonce = generateNonce()

    const requestBody = {
      deep_search: '0',
      req_id: reqId,
      model: actualModel,
      scene: 'chat',
      session_id: sessionId,
      sub_scene: 'chat',
      temporary: false,
      messages: [
        {
          content: finalContent,
          mime_type: 'text/plain',
          meta_data: {
            ori_query: finalContent
          }
        }
      ],
      from: 'default',
      parent_req_id: '0',
      biz_data: '{"entryPoint":"tongyigw"}',
      scene_param: request.session_id ? 'follow_up' : 'first_turn',
      chat_client: 'h5',
      client_tm: timestamp.toString(),
      protocol_version: 'v2',
      biz_id: 'ai_qwen'
    }

    const queryString = `biz_id=ai_qwen&chat_client=h5&device=pc&fr=pc&pr=qwen&ut=${uuid(false)}&nonce=${nonce}&timestamp=${timestamp}`
    const url = `${QWEN_API_BASE}/api/v2/chat?${queryString}`

    const response = await this.axiosInstance.post(url, requestBody, {
      headers: {
        ...DEFAULT_HEADERS,
        'Content-Type': 'application/json',
        Cookie: `tongyi_sso_ticket=${ticket}`,
      },
      responseType: 'stream',
      timeout: 120000,
      decompress: false,
    })

    return { response, sessionId, reqId }
  }

  async deleteSession(sessionId: string): Promise<boolean> {
    try {
      const ticket = this.getTicket()
      if (!ticket || !sessionId) {
        return false
      }

      const response = await axios.post(
        `${QWEN_API_BASE}/api/v2/session/delete`,
        { session_id: sessionId },
        {
          headers: {
            Cookie: `tongyi_sso_ticket=${ticket}`,
            ...DEFAULT_HEADERS,
            'X-Platform': 'pc_tongyi',
            'X-DeviceId': '5b68c267-cd8e-fd0e-148a-18345bc9a104',
          },
          params: {
            biz_id: 'ai_qwen',
            chat_client: 'h5',
            device: 'pc',
            fr: 'pc',
            pr: 'qwen',
            ut: '5b68c267-cd8e-fd0e-148a-18345bc9a104',
          },
          timeout: 15000,
          validateStatus: () => true,
        }
      )

      if (response.status !== 200) {
        console.warn(`[Qwen] Failed to delete session ${sessionId}: status ${response.status}`)
        return false
      }

      const { success, errorMsg } = response.data
      if (success === false) {
        console.warn(`[Qwen] Failed to delete session ${sessionId}: ${errorMsg}`)
        return false
      }

      return true
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error'
      console.warn('[Qwen] Failed to delete session:', errorMessage)
      return false
    }
  }

  static isQwenProvider(provider: Provider): boolean {
    return provider.id === 'qwen' || provider.apiEndpoint.includes('qianwen.com') || provider.apiEndpoint.includes('aliyun.com')
  }
}

export class QwenStreamHandler {
  private sessionId: string = ''
  private model: string
  private created: number
  private onEnd?: (sessionId: string) => void
  private content: string = ''
  private responseId: string = ''
  private stopSent: boolean = false
  private toolCallsSent: boolean = false

  constructor(model: string, onEnd?: (sessionId: string) => void) {
    this.model = model
    this.created = Math.floor(Date.now() / 1000)
    this.onEnd = onEnd
  }

  private sendToolCalls(transStream: PassThrough): void {
    if (this.toolCallsSent) return
    
    const toolCalls = parseToolUse(this.content)
    if (toolCalls && toolCalls.length > 0) {
      this.toolCallsSent = true
      
      // Send tool_calls delta
      for (let i = 0; i < toolCalls.length; i++) {
        const tc = toolCalls[i]
        transStream.write(
          `data: ${JSON.stringify({
            id: this.responseId || this.sessionId,
            model: this.model,
            object: 'chat.completion.chunk',
            choices: [{
              index: 0,
              delta: {
                tool_calls: [{
                  index: i,
                  id: tc.id,
                  type: 'function',
                  function: {
                    name: tc.function.name,
                    arguments: tc.function.arguments,
                  },
                }],
              },
              finish_reason: null,
            }],
            created: this.created,
          })}\n\n`
        )
      }
      
      // Send finish with tool_calls
      transStream.write(
        `data: ${JSON.stringify({
          id: this.responseId || this.sessionId,
          model: this.model,
          object: 'chat.completion.chunk',
          choices: [{ index: 0, delta: {}, finish_reason: 'tool_calls' }],
          usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
          created: this.created,
        })}\n\n`
      )
      transStream.end('data: [DONE]\n\n')
      this.onEnd?.(this.sessionId)
    }
  }

  handleStream(stream: any, response?: AxiosResponse): PassThrough {
    const transStream = new PassThrough()

    const contentEncoding = response?.headers?.['content-encoding']

    transStream.write(
      `data: ${JSON.stringify({
        id: '',
        model: this.model,
        object: 'chat.completion.chunk',
        choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }],
        created: this.created,
      })}\n\n`
    )

    let buffer = ''

    const processBuffer = () => {
      while (true) {
        const doubleNewlineIndex = buffer.indexOf('\n\n')
        if (doubleNewlineIndex === -1) break

        const eventBlock = buffer.substring(0, doubleNewlineIndex)
        buffer = buffer.substring(doubleNewlineIndex + 2)

        const lines = eventBlock.split('\n')
        let eventType = 'message'
        let eventData = ''

        for (const line of lines) {
          if (line.startsWith('event:')) {
            eventType = line.substring(6).trim()
          } else if (line.startsWith('data:')) {
            eventData = line.substring(5)
          }
        }

        if (eventData && eventData !== '[DONE]') {
          try {
            const result = JSON.parse(eventData)
            if (result.communication) {
              if (!this.sessionId && result.communication.sessionid) {
                this.sessionId = result.communication.sessionid
              }
              if (!this.responseId && result.communication.reqid) {
                this.responseId = result.communication.reqid
              }
            }

            if (result.data?.messages) {
              for (const msg of result.data.messages) {
                if ((msg.mime_type === 'text/plain' || msg.mime_type === 'multi_load/iframe') && msg.content) {
                  const newContent = msg.content
                  if (newContent.length > this.content.length) {
                    const chunk = newContent.substring(this.content.length)
                    this.content = newContent

                    transStream.write(
                      `data: ${JSON.stringify({
                        id: this.responseId || this.sessionId,
                        model: this.model,
                        object: 'chat.completion.chunk',
                        choices: [{ index: 0, delta: { content: chunk }, finish_reason: null }],
                        created: this.created,
                      })}\n\n`
                    )
                  }
                }

                if (msg.status === 'complete' || msg.status === 'finished') {
                  // 只有当 multi_load/iframe 消息完成时才发送 stop
                  if (msg.mime_type === 'multi_load/iframe' && !this.stopSent) {
                    this.stopSent = true
                    // Check for tool calls before sending stop
                    if (hasToolUse(this.content)) {
                      this.sendToolCalls(transStream)
                      return
                    }
                    
                    transStream.write(
                      `data: ${JSON.stringify({
                        id: this.responseId || this.sessionId,
                        model: this.model,
                        object: 'chat.completion.chunk',
                        choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
                        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
                        created: this.created,
                      })}\n\n`
                    )
                    transStream.end('data: [DONE]\n\n')
                    this.onEnd?.(this.sessionId)
                  }
                }
              }
            }

            if (result.error_code && result.error_code !== 0) {
              console.error('[Qwen] API error:', result.error_code, result.error_msg)
              transStream.write(
                `data: ${JSON.stringify({
                  id: this.responseId || this.sessionId,
                  model: this.model,
                  object: 'chat.completion.chunk',
                  choices: [{ index: 0, delta: { content: `\n[Error: ${result.error_msg || result.error_code}]` }, finish_reason: 'stop' }],
                  created: this.created,
                })}\n\n`
              )
              transStream.end('data: [DONE]\n\n')
            }
          } catch (err) {
            console.error('[Qwen] Parse error:', err, 'Data:', eventData.substring(0, 200))
          }
        }

        if (eventType === 'complete') {
          if (!transStream.closed && !this.stopSent) {
            this.stopSent = true
            
            // Check for tool calls before sending stop
            if (hasToolUse(this.content)) {
              this.sendToolCalls(transStream)
              return
            }
            
            transStream.write(
              `data: ${JSON.stringify({
                id: this.responseId || this.sessionId,
                model: this.model,
                object: 'chat.completion.chunk',
                choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
                usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
                created: this.created,
              })}\n\n`
            )
            transStream.end('data: [DONE]\n\n')
          }
        }
      }
    }

    let decompressStream: any = stream
    
    if (contentEncoding === 'gzip') {
      decompressStream = stream.pipe(createGunzip())
    } else if (contentEncoding === 'deflate') {
      decompressStream = stream.pipe(createInflate())
    } else if (contentEncoding === 'br') {
      decompressStream = stream.pipe(createBrotliDecompress())
    } else if (contentEncoding === 'zstd') {
      const chunks: Buffer[] = []
      stream.on('data', (chunk: Buffer) => chunks.push(chunk))
      stream.once('end', () => {
        try {
          const compressedData = Buffer.concat(chunks)
          ZstdCodec.run((zstd) => {
            const simple = new zstd.Simple()
            const decompressed = simple.decompress(compressedData)
            const decompressedStr = Buffer.from(decompressed).toString('utf8')
            buffer = decompressedStr
            processBuffer()
            transStream.end('data: [DONE]\n\n')
          })
        } catch (err) {
          console.error('[Qwen] Zstd decompression error:', err)
          transStream.end('data: [DONE]\n\n')
        }
      })
      stream.once('error', (err: Error) => {
        console.error('[Qwen] Stream error:', err)
        transStream.end('data: [DONE]\n\n')
      })
      return transStream
    }

    decompressStream.on('data', (bufferChunk: Buffer) => {
      buffer += bufferChunk.toString()
      processBuffer()
    })
    decompressStream.once('error', (err: Error) => {
      console.error('[Qwen] Stream error:', err)
      transStream.end('data: [DONE]\n\n')
    })
    decompressStream.once('close', () => {
      processBuffer()
      transStream.end('data: [DONE]\n\n')
    })

    return transStream
  }

  async handleNonStream(stream: any, response?: AxiosResponse): Promise<any> {
    return new Promise((resolve, reject) => {
      const data = {
        id: '',
        model: this.model,
        object: 'chat.completion',
        choices: [
          {
            index: 0,
            message: { role: 'assistant', content: '' },
            finish_reason: 'stop',
          },
        ],
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
        created: this.created,
      }

      let contentAccumulator = ''
      let buffer = ''
      let resolved = false

      const processBuffer = () => {
        while (true) {
          const doubleNewlineIndex = buffer.indexOf('\n\n')
          if (doubleNewlineIndex === -1) break

          const eventBlock = buffer.substring(0, doubleNewlineIndex)
          buffer = buffer.substring(doubleNewlineIndex + 2)

          const lines = eventBlock.split('\n')
          let eventType = 'message'
          let eventData = ''

          for (const line of lines) {
            if (line.startsWith('event:')) {
              eventType = line.substring(6).trim()
            } else if (line.startsWith('data:')) {
              eventData = line.substring(5)
            }
          }

          if (eventData && eventData !== '[DONE]') {
            try {
              const result = JSON.parse(eventData)
              if (result.communication) {
                if (!data.id && result.communication.sessionid) {
                  data.id = result.communication.sessionid
                  this.sessionId = result.communication.sessionid
                }
              }

              if (result.data?.messages) {
                for (const msg of result.data.messages) {
                  // Handle multi_load/iframe content (actual response content)
                  if (msg.mime_type === 'multi_load/iframe' && msg.content) {
                    contentAccumulator = msg.content
                  }
                  
                  // Also handle text/plain content
                  if (msg.mime_type === 'text/plain' && msg.content) {
                    if (msg.content.length > contentAccumulator.length) {
                      contentAccumulator = msg.content
                    }
                  }

                  if (msg.status === 'complete' || msg.status === 'finished') {
                    if (msg.mime_type === 'multi_load/iframe') {
                      data.choices[0].message.content = contentAccumulator
                      this.content = contentAccumulator
                      this.onEnd?.(this.sessionId)
                      resolved = true
                      resolve(data)
                      return
                    }
                  }
                }
              }
            } catch (err) {
              console.error('[Qwen] Non-stream parse error:', err)
            }
          }

          if (eventType === 'complete' && !resolved) {
            data.choices[0].message.content = contentAccumulator
            this.content = contentAccumulator
            resolved = true
            resolve(data)
            return
          }
        }
      }

      let decompressStream: any = stream
      
      const contentEncoding = response?.headers?.['content-encoding']?.toLowerCase()
      if (contentEncoding === 'gzip') {
        decompressStream = stream.pipe(createGunzip())
      } else if (contentEncoding === 'deflate') {
        decompressStream = stream.pipe(createInflate())
      } else if (contentEncoding === 'br') {
        decompressStream = stream.pipe(createBrotliDecompress())
      } else if (contentEncoding === 'zstd') {
        const chunks: Buffer[] = []
        stream.on('data', (chunk: Buffer) => chunks.push(chunk))
        stream.once('end', () => {
          try {
            const compressedData = Buffer.concat(chunks)
            ZstdCodec.run((zstd) => {
              const simple = new zstd.Simple()
              const decompressed = simple.decompress(compressedData)
              const decompressedStr = Buffer.from(decompressed).toString('utf8')
              buffer = decompressedStr
              processBuffer()
              data.choices[0].message.content = contentAccumulator
              this.content = contentAccumulator
              resolve(data)
            })
          } catch (err) {
            console.error('[Qwen] Zstd decompression error:', err)
            reject(err)
          }
        })
        stream.once('error', (err: Error) => {
          console.error('[Qwen] Non-stream error:', err)
          reject(err)
        })
        return
      }

      decompressStream.on('data', (chunk: Buffer) => {
        buffer += chunk.toString()
        processBuffer()
      })
      decompressStream.once('error', (err: Error) => {
        console.error('[Qwen] Non-stream error:', err)
        reject(err)
      })
      decompressStream.once('close', () => {
        if (!resolved) {
          processBuffer()
          data.choices[0].message.content = contentAccumulator
          this.content = contentAccumulator
          resolve(data)
        }
      })
      decompressStream.once('end', () => {
        if (!resolved) {
          processBuffer()
          data.choices[0].message.content = contentAccumulator
          this.content = contentAccumulator
          resolve(data)
        }
      })
    })
  }

  getSessionId(): string {
    return this.sessionId
  }
}

export const qwenAdapter = {
  QwenAdapter,
  QwenStreamHandler,
}
