import React, { Component } from 'react';
import './App.css'
import 'bootstrap/dist/css/bootstrap.min.css';
import axios from 'axios';

export default class App extends Component {
  constructor(props) {
    super(props)
    this.state = {
      file: null,
      image: null,
      file_name: null,
      similar_file_arr: null,
      imagesVisible: false,
    }
    this.uploadSingleFile = this.uploadSingleFile.bind(this)
    this.upload = this.upload.bind(this)
  }
  uploadSingleFile(e) {
    this.setState({
      file: URL.createObjectURL(e.target.files[0]),
      image: e.target.files[0]
    });
  }
  upload(e) {
    e.preventDefault()
    console.log(this.state.file)
  }
  convertToBase64(file) {
    return new Promise((resolve, reject) => {
      const fileReader = new FileReader();
      fileReader.readAsDataURL(file);
      fileReader.onload = () => {
        resolve(fileReader.result);
      };
      fileReader.onerror = (error) => {
        reject(error);
      };
    });
  };

  handleSubmit = (e) => {
    e.preventDefault();
    let formData = new FormData();
    formData.append('files', this.state.image)
    let url = 'http://localhost:8000/api/upload/';
    axios.post(url, formData, {
      headers: {
        'content-type': 'multipart/form-data'
      }
    })
      .then(res => {
        this.setState({
          file_name: res.data.file_name,
          similar_file_arr: res.data.results
        })
      })
      .catch(err => console.log(err))
    console.log(this.state.similar_file_arr);
  };

  handlePrediction = (e) => {
    e.preventDefault()
    this.setState({ imagesVisible: true });
  };

  render() {
    let imgPreview;
    if (this.state.file) {
      imgPreview = <img style={{ width: 250, height: 250 }} src={this.state.file} alt='' />;
    }
    let data = console.log(this.state);
    return (
      <div className="App">
        <form id="form-id" onSubmit={this.handleSubmit} >
          <h1>Reverse Image Search</h1>
          {imgPreview}
          <div className="form-group">
            <input type="file" accept="image/*" className="form-control" onChange={this.uploadSingleFile} />
          </div>
          {data}
          <input type="submit" value="Upload" className="btn btn-primary btn-block" />
        </form >
        <div>
          <form onSubmit={this.handlePrediction}>
            <button type="submit">Predict</button>
            {this.state.imagesVisible && (
              <div className="image-container">
                <h2>Similar Images Lookup</h2>
                {this.state.similar_file_arr.map((filePath, index) => (
                  <img style={{ width: 250, height: 250 }} key={index} src={filePath.replace(/\\/g, '/').replace(/.*\/public\//, './')} alt={`Image ${index}`} />
                ))}
              </div>
            )}
          </form>
        </div>
      </div>
    )
  }
}