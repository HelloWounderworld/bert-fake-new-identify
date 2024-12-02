console.log("Entrei no arquivo testeflask2 - Requisicao GET!");
fetch('http://localhost:5000/hello')
    .then(response => response.json())
    .then(data => {
        console.log("Deu certo! Verifique se devolveu a resposta 200!");
        console.log("Valor data: ", data);
    })
    .catch(error => console.error('Error fetching data:', error));