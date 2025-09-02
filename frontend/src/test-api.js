// Simple test to verify API service is working
// Run this in the browser console to test

import { api } from './services/api.js';

// Test API functions
async function testAPI() {
  console.log('Testing API service...');
  
  try {
    // Test webcam status
    console.log('1. Testing webcam status...');
    const status = await api.getWebcamStatus();
    console.log('Webcam status:', status);
    
    // Test login
    console.log('2. Testing login...');
    const loginResult = await api.login('admin', 'password123');
    console.log('Login result:', loginResult);
    
    console.log('✅ API service is working correctly!');
  } catch (error) {
    console.error('❌ API test failed:', error);
  }
}

// Export for use in browser console
window.testAPI = testAPI; 