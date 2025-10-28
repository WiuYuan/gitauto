const WS_PORT = import.meta.env.VITE_WS_PORT || 17860
const WS_URL = `ws://127.0.0.1:${WS_PORT}`

class WebSocketManager {
  constructor(options = {}) {
    this.options = {
      url: WS_URL,
      reconnectInterval: 3000,
      maxReconnectAttempts: 5,
      onMessage: () => {},
      onError: () => {},
      onOpen: () => {},
      onClose: () => {},
      ...options
    }
    
    this.ws = null
    this.reconnectAttempts = 0
    this.isConnected = false
    this.shouldReconnect = true
  }

  connect() {
    try {
      this.ws = new WebSocket(this.options.url)
      
      this.ws.onopen = () => {
        console.log('WebSocket connected')
        this.isConnected = true
        this.reconnectAttempts = 0
        this.options.onOpen()
        this.options.onMessage({
          type: 'status',
          data: 'connected'
        })
      }

      this.ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data)
          this.options.onMessage(message)
        } catch (error) {
          console.error('Error parsing WebSocket message:', error)
          // Handle raw HTML updates for backward compatibility
          if (typeof event.data === 'string') {
            this.options.onMessage({
              type: 'html_update',
              data: event.data
            })
          }
        }
      }

      this.ws.onclose = (event) => {
        console.log('WebSocket disconnected:', event.code, event.reason)
        this.isConnected = false
        this.options.onClose(event)
        this.options.onMessage({
          type: 'status',
          data: 'disconnected'
        })
        
        if (this.shouldReconnect && this.reconnectAttempts < this.options.maxReconnectAttempts) {
          setTimeout(() => {
            this.reconnectAttempts++
            console.log(`Attempting to reconnect... (${this.reconnectAttempts}/${this.options.maxReconnectAttempts})`)
            this.connect()
          }, this.options.reconnectInterval)
        }
      }

      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error)
        this.options.onError(error)
      }

    } catch (error) {
      console.error('Failed to create WebSocket connection:', error)
      this.options.onError(error)
    }
  }

  sendMessage(message) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message))
      return true
    } else {
      console.warn('WebSocket not connected, message not sent:', message)
      return false
    }
  }

  disconnect() {
    this.shouldReconnect = false
    if (this.ws) {
      this.ws.close()
      this.ws = null
    }
  }

  getConnectionStatus() {
    return this.isConnected ? 'connected' : 'disconnected'
  }
}

export default WebSocketManager