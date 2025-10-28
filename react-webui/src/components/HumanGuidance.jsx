import React, { useState } from 'react'

const HumanGuidance = ({ onSendGuidance }) => {
  const [guidanceText, setGuidanceText] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)

  const handleSubmit = () => {
    if (guidanceText.trim() === '') {
      alert('Please enter some guidance text')
      return
    }

    setIsSubmitting(true)
    
    // Send the guidance through the callback
    onSendGuidance({
      type: 'human_guidance',
      data: {
        content: guidanceText.trim()
      }
    })

    // Reset the form
    setGuidanceText('')
    setIsSubmitting(false)
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && e.ctrlKey) {
      handleSubmit()
    }
  }

  return (
    <div
      style={{
        border: '2px solid #ff6b6b',
        borderRadius: '8px',
        padding: '16px',
        margin: '16px 0',
        backgroundColor: '#1a1a1a',
        color: '#fff',
        fontFamily: 'Consolas, Menlo, monospace'
      }}
    >
      <h3 style={{ color: '#ff6b6b', marginBottom: '16px', fontSize: '1.2em' }}>
        💡 Human Guidance
      </h3>
      
      <div style={{ marginBottom: '16px' }}>
        <textarea
          value={guidanceText}
          onChange={(e) => setGuidanceText(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Enter your guidance here... (Ctrl+Enter to send)"
          style={{
            width: '100%',
            minHeight: '80px',
            padding: '8px',
            border: '1px solid #555',
            borderRadius: '4px',
            backgroundColor: '#2a2a2a',
            color: '#fff',
            fontFamily: 'inherit',
            fontSize: '14px',
            resize: 'vertical'
          }}
        />
      </div>

      <button
        onClick={handleSubmit}
        disabled={isSubmitting || guidanceText.trim() === ''}
        style={{
          padding: '8px 16px',
          border: 'none',
          borderRadius: '4px',
          backgroundColor: guidanceText.trim() === '' ? '#555' : '#ff6b6b',
          color: '#fff',
          cursor: guidanceText.trim() === '' ? 'not-allowed' : 'pointer',
          fontWeight: 'bold',
          transition: 'background-color 0.2s ease'
        }}
      >
        {isSubmitting ? 'Sending...' : 'Send Guidance'}
      </button>

      <div style={{ color: '#777', fontSize: '0.9em', marginTop: '8px' }}>
        Tip: Press Ctrl+Enter to quickly send your guidance
      </div>
    </div>
  )
}

export default HumanGuidance