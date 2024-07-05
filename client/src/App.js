// src/App.js
import React from 'react';
import './App.css';
import UploadForm from './components/UploadForm';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>DermDetect AI</h1>
      </header>
      <UploadForm />
    </div>
  );
}

export default App;

