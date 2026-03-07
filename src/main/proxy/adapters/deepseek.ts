/**
 * DeepSeek Adapter
 * Implements DeepSeek web API protocol
 */

import axios, { AxiosResponse } from 'axios'
import { getDeepSeekHash } from '../../lib/challenge'
import { Account, Provider } from '../store/types'
import { storeManager } from '../store/store'

const DEEPSEEK_API_BASE = 'https://chat.deepseek.com/api'

const FAKE_HEADERS = {
  Accept: '*/*',
  'Accept-Encoding': 'gzip, deflate, br, zstd',
  'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
  Origin: 'https://chat.deepseek.com',
  Referer: 'https://chat.deepseek.com/',
  'Sec-Ch-Ua': '"Chromium";v="134", "Not:A-Brand";v="24", "Google Chrome";v="134"',
  'Sec-Ch-Ua-Mobile': '?0',
  'Sec-Ch-Ua-Platform': '"macOS"',
  'Sec-Fetch-Dest': 'empty',
  'Sec-Fetch-Mode': 'cors',
  'Sec-Fetch-Site': 'same-origin',
  'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36',
  'X-App-Version': '20241129.1',
  'X-Client-Locale': 'zh-CN',
  'X-Client-Platform': 'web',
  'X-Client-Version': '1.6.1',
}

interface TokenInfo {
  accessToken: string
  refreshToken: string
  expiresAt: number
}

interface ChallengeResponse {
  algorithm: string
  challenge: string
  salt: string
  difficulty: number
  expire_at: number
  signature: string
}

interface DeepSeekMessage {
  role: 'user' | 'assistant' | 'system' | 'tool'
  content: string | any[] | null
  tool_call_id?: string
  tool_calls?: any[]
}

interface ChatCompletionRequest {
  model: string
  messages: DeepSeekMessage[]
  stream?: boolean
  temperature?: number
  web_search?: boolean
  reasoning_effort?: 'low' | 'medium' | 'high'
  tools?: any[]
  tool_choice?: any
  sessionId?: string
  parentMessageId?: string
}

const tokenCache = new Map<string, TokenInfo>()
const sessionCache = new Map<string, { sessionId: string; createdAt: number }>()

function generateRandomString(length: number, charset: string = 'alphanumeric'): string {
  const sets = {
    numeric: '0123456789',
    alphabetic: 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
    alphanumeric: '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
    hex: '0123456789abcdef',
  }
  const chars = sets[charset as keyof typeof sets] || sets.alphanumeric
  let result = ''
  for (let i = 0; i < length; i++) {
    result += chars.charAt(Math.floor(Math.random() * chars.length))
  }
  return result
}

function uuid(): string {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0
    const v = c === 'x' ? r : (r & 0x3) | 0x8
    return v.toString(16)
  })
}

function generateCookie(): string {
  const timestamp = Date.now()
  return `intercom-HWWAFSESTIME=${timestamp}; HWWAFSESID=${generateRandomString(18, 'hex')}; Hm_lvt_${uuid(false)}=${Math.floor(timestamp / 1000)},${Math.floor(timestamp / 1000)},${Math.floor(timestamp / 1000)}; Hm_lpvt_${uuid(false)}=${Math.floor(timestamp / 1000)}; _frid=${uuid(false)}; _fr_ssid=${uuid(false)}; _fr_pvid=${uuid(false)}`
}

function unixTimestamp(): number {
  return Math.floor(Date.now() / 1000)
}

export class DeepSeekAdapter {
  private provider: Provider
  private account: Account
  private token: string

  constructor(provider: Provider, account: Account) {
    this.provider = provider
    this.account = account
    console.log('[DeepSeek] Account credentials:', JSON.stringify(account.credentials, null, 2))
    this.token = account.credentials.token || account.credentials.apiKey || account.credentials.refreshToken || ''
    console.log('[DeepSeek] Using token:', this.token.substring(0, 20) + '...')
  }

  private async acquireToken(): Promise<string> {
    if (!this.token) {
      throw new Error('DeepSeek Token not configured, please add Token in account settings')
    }

    const cached = tokenCache.get(this.token)
    if (cached && cached.expiresAt > unixTimestamp()) {
      return cached.accessToken
    }

    console.log('[DeepSeek] Acquiring token...')
    
    const result = await axios.get(`${DEEPSEEK_API_BASE}/v0/users/current`, {
      headers: {
        Authorization: `Bearer ${this.token}`,
        ...FAKE_HEADERS,
      },
      timeout: 15000,
      validateStatus: () => true,
    })

    console.log('[DeepSeek] Token response status:', result.status)
    
    if (result.status === 401 || result.status === 403) {
      throw new Error(`Token invalid or expired, please get a new Token`)
    }

    if (result.status !== 200) {
      throw new Error(`Failed to acquire token: HTTP ${result.status}`)
    }

    // Response structure: { code: 0, data: { biz_code: 0, biz_data: { token: "..." } } }
    const bizData = result.data?.data?.biz_data || result.data?.biz_data
    if (!bizData?.token) {
      const errorMsg = result.data?.msg || result.data?.data?.biz_msg || 'Unknown error'
      console.log('[DeepSeek] Token response data:', JSON.stringify(result.data, null, 2))
      throw new Error(`Failed to acquire token: ${errorMsg}`)
    }

    const accessToken = bizData.token
    tokenCache.set(this.token, {
      accessToken,
      refreshToken: this.token,
      expiresAt: unixTimestamp() + 3600,
    })

    console.log('[DeepSeek] Token acquired successfully')
    return accessToken
  }

  private async createSession(): Promise<string> {
    const cacheKey = this.account.id
    const cached = sessionCache.get(cacheKey)
    if (cached && Date.now() - cached.createdAt < 300000) {
      return cached.sessionId
    }

    const token = await this.acquireToken()
    const result = await axios.post(
      `${DEEPSEEK_API_BASE}/v0/chat_session/create`,
      { character_id: null },
      {
        headers: {
          Authorization: `Bearer ${token}`,
          ...FAKE_HEADERS,
          Cookie: generateCookie(),
        },
        timeout: 15000,
        validateStatus: () => true,
      }
    )

    console.log('[DeepSeek] Create session response:', JSON.stringify(result.data, null, 2))

    // Response structure: { code: 0, data: { biz_code: 0, biz_data: { id: "..." } } }
    const bizData = result.data?.data?.biz_data || result.data?.biz_data
    if (result.status !== 200 || !bizData?.id) {
      throw new Error(`Failed to create session: ${result.data?.msg || result.data?.data?.biz_msg || result.status}`)
    }

    const sessionId = bizData.id
    sessionCache.set(cacheKey, { sessionId, createdAt: Date.now() })

    return sessionId
  }

  async deleteSession(sessionId: string): Promise<boolean> {
    try {
      const token = await this.acquireToken()
      const result = await axios.post(
        `${DEEPSEEK_API_BASE}/v0/chat_session/delete`,
        { chat_session_id: sessionId },
        {
          headers: {
            Authorization: `Bearer ${token}`,
            ...FAKE_HEADERS,
          },
          timeout: 15000,
          validateStatus: () => true,
        }
      )

      console.log('[DeepSeek] Delete session response:', JSON.stringify(result.data, null, 2))

      const success = result.status === 200 && result.data?.code === 0
      if (success) {
        // Clear cache
        const cacheKey = this.account.id
        sessionCache.delete(cacheKey)
        console.log('[DeepSeek] Session deleted:', sessionId)
      }
      return success
    } catch (error) {
      console.error('[DeepSeek] Failed to delete session:', error)
      return false
    }
  }

  private async getChallenge(targetPath: string): Promise<ChallengeResponse> {
    const token = await this.acquireToken()
    const result = await axios.post(
      `${DEEPSEEK_API_BASE}/v0/chat/create_pow_challenge`,
      { target_path: targetPath },
      {
        headers: {
          Authorization: `Bearer ${token}`,
          ...FAKE_HEADERS,
        },
        timeout: 15000,
        validateStatus: () => true,
      }
    )

    // Response structure: { code: 0, data: { biz_code: 0, biz_data: { challenge: {...} } } }
    const bizData = result.data?.data?.biz_data || result.data?.biz_data
    if (result.status !== 200 || !bizData?.challenge) {
      throw new Error(`Failed to get challenge: ${result.data?.msg || result.data?.data?.biz_msg || result.status}`)
    }

    return bizData.challenge
  }

  private async calculateChallengeAnswer(challenge: ChallengeResponse): Promise<string> {
    const { algorithm, challenge: challengeStr, salt, difficulty, expire_at, signature } = challenge
    
    if (algorithm !== 'DeepSeekHashV1') {
      throw new Error(`Unsupported algorithm: ${algorithm}`)
    }
    
    console.log('[DeepSeek] Challenge parameters:', { difficulty })
    
    const deepSeekHash = await getDeepSeekHash()
    const answer = deepSeekHash.calculateHash(algorithm, challengeStr, salt, difficulty, expire_at)
    
    if (answer === undefined) {
      throw new Error('Challenge calculation failed')
    }
    
    console.log('[DeepSeek] Challenge answer found:', answer)

    return Buffer.from(JSON.stringify({
      algorithm,
      challenge: challengeStr,
      salt,
      answer,
      signature,
      target_path: '/api/v0/chat/completion',
    })).toString('base64')
  }


  private stringifyContent(content: DeepSeekMessage['content']): string {
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

  private escapeXml(text: string): string {
    return text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
  }

  private normalizeMessages(messages: DeepSeekMessage[]): DeepSeekMessage[] {
    const toolNameByCallId = new Map<string, string>()
    const normalized: DeepSeekMessage[] = []

    for (const msg of messages) {
      if (msg.role === 'assistant' && Array.isArray(msg.tool_calls) && msg.tool_calls.length > 0) {
        const toolUseBlocks: string[] = []
        for (const tc of msg.tool_calls) {
          const callId = tc?.id || ''
          const toolName = tc?.function?.name || 'tool'
          const args = tc?.function?.arguments || '{}'
          if (callId) {
            toolNameByCallId.set(callId, toolName)
          }
          toolUseBlocks.push(
            `<tool_use>\n<name>${this.escapeXml(toolName)}</name>\n<arguments>${this.escapeXml(args)}</arguments>\n</tool_use>`
          )
        }
        normalized.push({
          ...msg,
          content: toolUseBlocks.join('\n'),
          tool_calls: undefined,
        })
        continue
      }

      if (msg.role === 'tool') {
        const toolName = (msg.tool_call_id && toolNameByCallId.get(msg.tool_call_id)) || 'tool'
        const resultText = this.stringifyContent(msg.content)
        normalized.push({
          role: 'user',
          content: `<tool_use_result>\n<name>${this.escapeXml(toolName)}</name>\n<result>${this.escapeXml(resultText)}</result>\n</tool_use_result>`,
        })
        continue
      }

      normalized.push({
        ...msg,
        content: this.stringifyContent(msg.content),
      })
    }

    return normalized
  }

  private messagesToPrompt(messages: DeepSeekMessage[]): string {
    const normalizedMessages = this.normalizeMessages(messages)
    const processedMessages = normalizedMessages.map((message) => ({
      role: message.role,
      text: this.stringifyContent(message.content),
    }))

    if (processedMessages.length === 0) return ''

    const mergedBlocks: { role: string; text: string }[] = []
    let currentBlock = { ...processedMessages[0] }

    for (let i = 1; i < processedMessages.length; i++) {
      const msg = processedMessages[i]
      if (msg.role === currentBlock.role) {
        currentBlock.text += `\n\n${msg.text}`
      } else {
        mergedBlocks.push(currentBlock)
        currentBlock = { ...msg }
      }
    }
    mergedBlocks.push(currentBlock)

    const userTag = '<\uFF5CUser\uFF5C>'
    const assistantTag = '<\uFF5CAssistant\uFF5C>'
    const assistantEndTag = '<\uFF5Cend of sentence\uFF5C>'

    return mergedBlocks
      .map((block, index) => {
        if (block.role === 'assistant') {
          return `${assistantTag}${block.text}${assistantEndTag}`
        }
        if (block.role === 'user' || block.role === 'system') {
          return index > 0 ? `${userTag}${block.text}` : block.text
        }
        if (block.role === 'tool') {
          return `${userTag}${block.text}`
        }
        return block.text
      })
      .join('')
      .replace(/!\[.+\]\(.+\)/g, '')
  }
  async chatCompletion(request: ChatCompletionRequest): Promise<{ response: AxiosResponse; sessionId: string; messageId: string }> {
    const token = await this.acquireToken()
    
    let sessionId = request.sessionId
    let parentMessageId = request.parentMessageId || null
    
    if (!sessionId) {
      sessionId = await this.createSession()
    }
    
    const challenge = await this.getChallenge('/api/v0/chat/completion')
    const challengeAnswer = await this.calculateChallengeAnswer(challenge)

    const messages = [...request.messages]

    const prompt = this.messagesToPrompt(messages)

    let searchEnabled = false
    let thinkingEnabled = false

    if (request.web_search) {
      searchEnabled = true
      console.log('[DeepSeek] Web search enabled')
    }

    if (request.reasoning_effort) {
      thinkingEnabled = true
      console.log('[DeepSeek] Reasoning mode enabled, effort:', request.reasoning_effort)
    }

    const modelLower = request.model.toLowerCase()
    if (!searchEnabled && modelLower.includes('search')) {
      searchEnabled = true
      console.log('[DeepSeek] Web search enabled (from model name)')
    }
    if (!thinkingEnabled && (modelLower.includes('r1') || modelLower.includes('think'))) {
      thinkingEnabled = true
      console.log('[DeepSeek] Reasoning mode enabled (from model name)')
    }
    if (!thinkingEnabled && prompt.includes('deep thinking')) {
      thinkingEnabled = true
      console.log('[DeepSeek] Reasoning mode enabled (from prompt)')
    }

    console.log('[DeepSeek] Using session:', { sessionId, parentMessageId })
    console.log('[DeepSeek] Prompt length:', prompt.length, 'chars')

    const response = await axios.post(
      `${DEEPSEEK_API_BASE}/v0/chat/completion`,
      {
        chat_session_id: sessionId,
        parent_message_id: parentMessageId,
        prompt,
        ref_file_ids: [],
        search_enabled: searchEnabled,
        thinking_enabled: thinkingEnabled,
      },
      {
        headers: {
          Authorization: `Bearer ${token}`,
          ...FAKE_HEADERS,
          Cookie: generateCookie(),
          'X-Ds-Pow-Response': challengeAnswer,
        },
        timeout: 120000,
        validateStatus: () => true,
        responseType: 'stream',
      }
    )

    return { response, sessionId, messageId: '' }
  }

  static isDeepSeekProvider(provider: Provider): boolean {
    return provider.id === 'deepseek' || provider.apiEndpoint.includes('deepseek.com')
  }
}

export const deepSeekAdapter = {
  DeepSeekAdapter,
}

