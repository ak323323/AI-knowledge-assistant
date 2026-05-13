const express = require("express");
const cors = require("cors");
const askRoute = require("./routes/ask");
const axios = require('axios');

const app = express();
app.use(cors());
app.use(express.json());

app.use("/ask", askRoute);

app.listen(3001, () => {
    console.log("API running on port 3001")
});