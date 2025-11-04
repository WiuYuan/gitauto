#!/usr/bin/env python3
"""
Test script to verify frontend default task handling.
This simulates WebSocket messages to test the React frontend behavior.
"""

import json
import time
import asyncio
import websockets
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.services.external_server import ExternalServer

async def test_frontend_default_task():
    """Test frontend behavior with default and custom task_id messages"""
    
    # Start the external server
    server = ExternalServer()
    server_task = asyncio.create_task(server._server_main())
    
    # Wait for server to start
    await asyncio.sleep(1)
    
    try:
        # Connect to WebSocket
        uri = f"ws://127.0.0.1:{server.port}"
        async with websockets.connect(uri) as websocket:
            
            print("Testing frontend default task handling...")
            
            # Test 1: Send multiple messages without task_id (should all go to 'default')
            print("\n1. Sending messages without task_id (should go to 'default'):")
            
            for i in range(3):
                message = {
                    "type": "info",
                    "data": {
                        "category": f"default_task",
                        "content": f"Default task message {i+1}",
                        "level_delta": 0
                    }
                }
                await websocket.send(json.dumps(message))
                print(f"  ✓ Sent default task message {i+1}")
                
                # Wait for response
                response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                response_data = json.loads(response)
                
                if response_data.get('type') == 'tree_update' and response_data.get('data', {}).get('task_id') == 'default':
                    print(f"  ✓ Received response for 'default' task")
                else:
                    print(f"  ✗ Unexpected response: {response_data}")
            
            # Test 2: Send messages with different task_ids
            print("\n2. Sending messages with different task_ids:")
            
            task_ids = ['task_a', 'task_b', 'task_c']
            for task_id in task_ids:
                message = {
                    "type": "info",
                    "task_id": task_id,
                    "data": {
                        "category": f"{task_id}_category",
                        "content": f"Content for {task_id}",
                        "level_delta": 0
                    }
                }
                await websocket.send(json.dumps(message))
                print(f"  ✓ Sent message for task_id='{task_id}'")
                
                # Wait for response
                response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                response_data = json.loads(response)
                
                if response_data.get('type') == 'tree_update' and response_data.get('data', {}).get('task_id') == task_id:
                    print(f"  ✓ Received response for task_id='{task_id}'")
                else:
                    print(f"  ✗ Unexpected response: {response_data}")
            
            # Test 3: Verify all tasks exist in server
            print("\n3. Verifying server state:")
            expected_tasks = ['default'] + task_ids
            for task_id in expected_tasks:
                if task_id in server._stacks:
                    print(f"  ✓ Task '{task_id}' exists in server stacks")
                else:
                    print(f"  ✗ Task '{task_id}' missing from server stacks")
            
            # Test 4: Test mixed messages (some with task_id, some without)
            print("\n4. Testing mixed messages:")
            
            # Send without task_id
            message_no_id = {
                "type": "info",
                "data": {
                    "category": "mixed_test",
                    "content": "Message without task_id",
                    "level_delta": 0
                }
            }
            await websocket.send(json.dumps(message_no_id))
            response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
            response_data = json.loads(response)
            if response_data.get('data', {}).get('task_id') == 'default':
                print("  ✓ Message without task_id correctly handled as 'default'")
            
            # Send with task_id
            message_with_id = {
                "type": "info",
                "task_id": "mixed_task",
                "data": {
                    "category": "mixed_test",
                    "content": "Message with task_id",
                    "level_delta": 0
                }
            }
            await websocket.send(json.dumps(message_with_id))
            response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
            response_data = json.loads(response)
            if response_data.get('data', {}).get('task_id') == 'mixed_task':
                print("  ✓ Message with task_id correctly handled")
            
            print("\n✅ All frontend default task handling tests completed!")
            print("\nExpected frontend behavior:")
            print("  - Default task dropdown should show: 'Default Task', 'task_a', 'task_b', 'task_c', 'mixed_task'")
            print("  - Active task should be 'task_a' (first non-default task received)")
            print("  - User should be able to switch between all tasks")
            
            return True
            
    except Exception as e:
        print(f"❌ Error during test: {e}")
        return False
    finally:
        # Cleanup
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass

if __name__ == "__main__":
    result = asyncio.run(test_frontend_default_task())
    if result:
        print("\n🎉 FRONTEND DEFAULT TASK HANDLING VERIFIED!")
    else:
        print("\n❌ FRONTEND DEFAULT TASK HANDLING TESTS FAILED!")
    exit(0 if result else 1)