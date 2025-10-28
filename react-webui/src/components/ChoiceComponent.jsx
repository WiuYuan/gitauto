import React, { useState, useEffect } from 'react'

const ChoiceComponent = ({ choiceData, onChoiceSelected }) => {
  const [selectedOption, setSelectedOption] = useState(null)
  const [isSubmitted, setIsSubmitted] = useState(false)

  useEffect(() => {
    // Reset state when new choice data arrives
    setSelectedOption(null)
    setIsSubmitted(false)
  }, [choiceData])

  const handleOptionSelect = (optionIndex) => {
    setSelectedOption(optionIndex)
    setIsSubmitted(true)
    onChoiceSelected({
      choiceId: choiceData.choiceId,
      selectedOption: optionIndex,
      optionText: choiceData.options[optionIndex]
    })
  }

  if (!choiceData || !choiceData.question || !choiceData.options) {
    return null
  }

  return (
    <div
      style={{
        border: '2px solid #ffd700',
        borderRadius: '8px',
        padding: '16px',
        margin: '16px 0',
        backgroundColor: '#1a1a1a',
        color: '#fff',
        fontFamily: 'Consolas, Menlo, monospace'
      }}
    >
      <h3 style={{ color: '#ffd700', marginBottom: '16px', fontSize: '1.2em' }}>
        🤔 {choiceData.question}
      </h3>
      
      <div style={{ marginBottom: '16px' }}>
        {choiceData.options.map((option, index) => (
          <div
            key={index}
            onClick={() => !isSubmitted && handleOptionSelect(index)}
            style={{
              padding: '8px 12px',
              margin: '4px 0',
              border: selectedOption === index ? '2px solid #00ff00' : '1px solid #555',
              borderRadius: '4px',
              backgroundColor: selectedOption === index ? '#2a2a2a' : 'transparent',
              cursor: isSubmitted ? 'default' : 'pointer',
              transition: 'all 0.2s ease',
              color: isSubmitted && selectedOption === index ? '#00ff00' : '#fff'
            }}
          >
            {String.fromCharCode(65 + index)}. {option}
          </div>
        ))}
      </div>

      {!isSubmitted ? (
        <div style={{ color: '#777', fontSize: '0.9em', marginTop: '8px' }}>
          Click an option to submit your choice
        </div>
      ) : (
        <div style={{ color: '#00ff00', fontWeight: 'bold' }}>
          ✅ Choice submitted: {String.fromCharCode(65 + selectedOption)}. {choiceData.options[selectedOption]}
        </div>
      )}
    </div>
  )
}

export default ChoiceComponent