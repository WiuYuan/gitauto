import React, { useState, useEffect, useRef } from 'react'
import TreeComponent from './components/TreeComponent'
import ChoiceComponent from './components/ChoiceComponent'
import HumanGuidance from './components/HumanGuidance'
import WebSocketManager from './services/WebSocketManager'
import './App.css'

function App() {
  const [taskTrees, setTaskTrees] = useState({}) // task_id -> treeData mapping
  const [connectionStatus, setConnectionStatus] = useState('disconnected')
  const [choiceData, setChoiceData] = useState(null)
  const [activeTaskId, setActiveTaskId] = useState('default') // Currently active task
  const wsManagerRef = useRef(null)

  useEffect(() => {
    // Initialize WebSocket connection
    wsManagerRef.current = new WebSocketManager({
      onMessage: (message) => {
        console.log('Received message:', message)
        // Handle different message types
        if (message.type === 'tree_update') {
          // Update taskTrees state using task_id as key
          const treeData = message.data
          const taskId = treeData.task_id || 'default'
          setTaskTrees(prev => ({
            ...prev,
            [taskId]: treeData
          }))
          // Set active task if not already set
          if (activeTaskId === 'default' && taskId !== 'default') {
            setActiveTaskId(taskId)
          }
        } else if (message.type === 'status') {
          setConnectionStatus(message.data)
        } else if (message.type === 'choice_request') {
          setChoiceData(message.data)
        }else if (message.type === 'clear_ack') {
          console.log('🧹 Received clear_ack:', message.data)
          // ✅ 清空任务树，只保留 default
          setTaskTrees({ default: { category: 'root', items: [], children: [] } })
          setActiveTaskId('default')
  
          // ✅ 可选：给出 UI 提示
          // alert('✅ All trees cleared, only default remains.')
        }
      },
      onError: (error) => {
        console.error('WebSocket error:', error)
        setConnectionStatus('error')
      }
    })

    // Connect to WebSocket
    wsManagerRef.current.connect()

    return () => {
      if (wsManagerRef.current) {
        wsManagerRef.current.disconnect()
      }
    }
  }, [])

  const handleSendTestMessage = () => {
    if (wsManagerRef.current) {
      wsManagerRef.current.sendMessage({
        type: 'test_message',
        content: 'Test message from React UI'
      })
    }
  }

  const handleClearTree = () => {
    if (wsManagerRef.current) {
      wsManagerRef.current.sendMessage({
        type: 'clear_tree',
        data: {}
      })
    }
  }

  const handleChoiceSelected = (choiceResult) => {
    if (wsManagerRef.current) {
      const msg = {
        type: 'choice_response',
        data: choiceResult
      }
      console.log('📤 Sending choice_response:', msg)
      wsManagerRef.current.sendMessage(msg)
      setChoiceData(null) // Clear choice data after submission
    }
  }
  const handleSendGuidance = (guidanceData) => {
    if (wsManagerRef.current) {
      console.log('📤 Sending human_guidance:', guidanceData)
      wsManagerRef.current.sendMessage(guidanceData)
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>🧠 React Logical Tree WebUI</h1>
        <div className="status-container">
          <div className={`status ${connectionStatus}`}>
            Status: {connectionStatus === 'connected' ? 'Connected' : 'Disconnected'}
          </div>
        </div>
      </header>

      <div className="controls">
        <button onClick={handleSendTestMessage}>
          Send Test Message
        </button>
        <button onClick={handleClearTree}>
          Clear Tree
        </button>
      </div>

      {/* Choice Component */}
      {choiceData && (
        <div className="choice-container">
          <ChoiceComponent 
            choiceData={choiceData} 
            onChoiceSelected={handleChoiceSelected} 
          />
        </div>
      )}

      {/* Task Selection UI */}
      {Object.keys(taskTrees).length > 0 && (
        <div className="task-selector">
          <label>Active Task: </label>
          <select 
            value={activeTaskId} 
            onChange={(e) => setActiveTaskId(e.target.value)}
            style={{ marginLeft: '10px', padding: '5px' }}
          >
            {Object.keys(taskTrees).map(taskId => (
              <option key={taskId} value={taskId}>
                {taskId === 'default' ? 'Default Task' : taskId}
              </option>
            ))}
          </select>
        </div>
      )}

      <div className="tree-container">
        {taskTrees[activeTaskId] && Object.keys(taskTrees[activeTaskId]).length > 0 ? (
          <TreeComponent data={taskTrees[activeTaskId]} />
        ) : (
          <p>No data yet...</p>
        )}
      </div>

      {/* Human Guidance Component */}
      <div className="guidance-container">
        <HumanGuidance onSendGuidance={handleSendGuidance} />
      </div>

      <footer className="app-footer">
        <p>Streaming WebUI - Real-time Logical Tree Display</p>
      </footer>
    </div>
  )
}

export default App