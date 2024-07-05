import React, { useState } from 'react';
import axios from 'axios';

const UploadForm = () => {
  const [file, setFile] = useState(null);
  const [message, setMessage] = useState('');
  const [prediction, setPrediction] = useState('');
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
  
    if (!file) {
      setMessage('Please select a file');
      return;
    }
  
    const formData = new FormData();
    formData.append('file', file);
  
    try {
      setLoading(true);
      setMessage('');
      setPrediction('');

      // Upload file
      const uploadResponse = await axios.post('http://localhost:5000/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      console.log('Upload Response:', uploadResponse.data);
      setMessage(uploadResponse.data.message);

      // If upload successful, trigger training and request prediction
      if (uploadResponse.status === 200) {
        // Trigger training if not already trained
        const trainResponse = await axios.post('http://localhost:5000/train');
        console.log('Train Response:', trainResponse.data);
        
        if (trainResponse.status === 200) {
          const predictResponse = await axios.post('http://localhost:5000/predict', formData, {
            headers: {
              'Content-Type': 'multipart/form-data'
            }
          });
          console.log('Prediction Response:', predictResponse.data);
          setPrediction(predictResponse.data.prediction);
        } else {
          setMessage('Error in training the model');
        }
      }
    } catch (error) {
      console.error('Error in form submission:', error);
      setMessage('Error uploading file');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="upload-form">
      <h2>Upload Image</h2>
      <form onSubmit={handleSubmit}>
        <input type="file" onChange={handleFileChange} />
        <button type="submit">Upload</button>
      </form>
      {loading && <p>Loading...</p>}
      {message && <p>{message}</p>}
      {prediction && <p>Prediction: {prediction}</p>}
    </div>
  );
};

export default UploadForm;
