async function runExample() {

    var x = new Float32Array( 1, 11 )

    var x = [];

     x[0] = document.getElementsByName('1').value;
     x[1] = document.getElementsByName('2').value;
     x[2] = document.getElementsByName('3').value;
     x[3] = document.getElementsByName('4').value;
     x[4] = document.getElementsByName('5').value;
     x[5] = document.getElementsByName('6').value;
     x[6] = document.getElementsByName('7').value;
     x[7] = document.getElementsByName('8').value;
     x[8] = document.getElementsByName('9').value;
     x[9] = document.getElementsByName('10').value;
     x[10] = document.getElementsByName('11').value;

    let tensorX = new onnx.Tensor(x, 'float32', [1, 11]);

    let session = new onnx.InferenceSession();

    await session.loadModel("./DLnet_WineData.onnx");
    let outputMap = await session.run([tensorX]);
    let outputData = outputMap.get('output1');

   let predictions = document.getElementById('predictions');

  predictions.innerHTML = ` <hr> Got an output tensor with values: <br/>
   <table>
     <tr>
       <td>  Red Wine Quality Rating  </td>
       <td id="td0">  ${outputData.data[0].toFixed(2)}  </td>
     </tr>
  </table>`;
    


}