const express = require("express");
const router = express.Router();
const axios = require("axios");

router.post("/", async (req, res) => {
  try {
    if (!req.body.question) {
      return res.status(400).json({ error: "Question is required" });
    }
      console.log("Incoming question:", req.body.question);

    const response = await axios.post("http://localhost:8000/ask", {
      question: req.body.question,
    });

    res.json(response.data); //
  } catch (err) {
    console.error("ERROR:", err.response?.data || err.message);

    res.status(500).json({
      error: err.response?.data || err.message,
    });
  }
});   

module.exports = router;