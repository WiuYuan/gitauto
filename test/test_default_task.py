#!/usr/bin/env python3
"""
Test script to verify default task handling when no task_id is provided.
This tests both backend and frontend behavior for tasks without task_id.
"""

import json
import time
import asyncio
import websockets
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.services.external_server import ExternalServer

async def test_default_task_handling():
    """Test that tasks without task_id are treated as 'default'"""
    
    # Start the external server
    server = ExternalServer()
    server_task = asyncio.create_task(server._server_main())
    
    # Wait for server to start
    await asyncio.sleep(1)
    
    try:
        # Connect to WebSocket
        uri = f"ws://127.0.0.1:{server.port}"
        async with websockets.connect(uri) as websocket:
            
            print("Testing default task handling...")
            
            # Test 1: Send message without task_id (should be treated as 'default')
            message_without_task_id = {
                "type": "info",
                "data": {
                    "category": "test_category",
                    "content": "Test content without task_id",
                    "level_delta": 0
                }
            }
            
            await websocket.send(json.dumps(message_without_task_id))
            print("✓ Sent message without task_id")
            
            # Wait for response
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            response_data = json.loads(response)
            
            print(f"Received response: {response_data}")
            
            # Verify the response has task_id = 'default'
            if response_data.get('type') == 'tree_update' and response_data.get('data', {}).get('task_id') == 'default':
                print("✓ SUCCESS: Message without task_id was treated as 'default'")
            else:
                print("✗ FAILED: Message without task_id was not treated as 'default'")
                return False
            
            # Test 2: Send message with explicit task_id
            message_with_task_id = {
                "type": "info",
                "task_id": "custom_task",
                "data": {
                    "category": "test_category",
                    "content": "Test content with custom_task",
                    "level_delta": 0
                }
            }
            
            await websocket.send(json.dumps(message_with_task_id))
            print("✓ Sent message with task_id='custom_task'")
            
            # Wait for response
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            response_data = json.loads(response)
            
            print(f"Received response: {response_data}")
            
            # Verify the response has task_id = 'custom_task'
            if response_data.get('type') == 'tree_update' and response_data.get('data', {}).get('task_id') == 'custom_task':
                print("✓ SUCCESS: Message with task_id='custom_task' was handled correctly")
            else:
                print("✗ FAILED: Message with task_id='custom_task' was not handled correctly")
                return False
            
            # Test 3: Verify both tasks exist in server's _stacks
            if 'default' in server._stacks and 'custom_task' in server._stacks:
                print("✓ SUCCESS: Both 'default' and 'custom_task' exist in server stacks")
                print(f"  - Default stack: {server._stacks['default'].to_dict()}")
                print(f"  - Custom task stack: {server._stacks['custom_task'].to_dict()}")
            else:
                print("✗ FAILED: Not all expected tasks exist in server stacks")
                return False
            
            return True
            
    except Exception as e:
        print(f"✗ Error during test: {e}")
        return False
    finally:
        # Cleanup
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass

if __name__ == "__main__":
    result = asyncio.run(test_default_task_handling())
    if result:
        print("\n🎉 ALL TESTS PASSED: Default task handling is working correctly!")
    else:
        print("\n❌ SOME TESTS FAILED: Default task handling needs attention!")
    exit(0 if result else 1)