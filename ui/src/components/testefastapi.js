//testefastapi
console.log("Hello WounderWorld! By JavaScript! Request to the FastAPI!");
async function fetchDataFastAPI(text) {
    console.log("Entrei na funcao fetchData");
    console.log("Show me the text: ", text);
    let msg = ''

    const data = await fetch('http://localhost:3000/analyzetext', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ string: text })
    });

    const responseData = await data.json();

    msg = responseData.message;
    
    return msg;
}

module.exports = { fetchDataFastAPI };