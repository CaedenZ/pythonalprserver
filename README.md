# pythonalprserver
## API func

```python
@app.route('/api/v1/newcamera/', methods=['POST'])
```

Take in camera,camModuleList,camSchedList object to create a scheduled process


```python
@app.route('/api/v1/getframe/', methods=['GET'])
```

Return one frame from the camera, take in ip as ipAdd

```python
@app.route('/api/v1/removecamera/', methods=['DELETE'])
```

Remove camera, take in ip as ipAdd
