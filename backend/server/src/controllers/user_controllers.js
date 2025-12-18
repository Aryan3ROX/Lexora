import pool from "../db_connection.js";

const FASTAPI_URL = "http://127.0.0.1/ai/recommendations";

const get_recommendations = async (req, res) => {
  try {
    const user_id = req.user.user_id;
    const response = await fetch(FASTAPI_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id }),
    });

    const data = await response.json();

    return res.status(response.status).json(data);
  } catch (e) {
    console.error("Proxy error:", e);
    return res.status(500).json({ error: "Internal server error" });
  }
};

const get_dashboard = async (req, res) => {
  const user_id = req.user.user_id;

  try {
    let results = await pool.query("SELECT * FROM users WHERE id = $1", [
      user_id,
    ]);
    const users_data = results.rows[0];
    results = await pool.query("SELECT * FROM user_books WHERE user_id = $1", [
      user_id,
    ]);
    const book_data = results.rows;

    results = {};
    results["member_since"] = users_data["created_at"];
    results["name"] = users_data["fullname"];
    results["username"] = req.user.username;
    results["books_viewed"] = [];
    results["ratings"] = {};
    results["statuses"] = {};

    await Promise.all(
      book_data.map(async (book) => {
        const id = book["book_id"];
        let book_details = await pool.query(
          "SELECT title, author, cover_image_url FROM books WHERE id = $1",
          [id]
        );
        book_details = book_details.rows;

        results["books_viewed"].push({
          title: book_details[0]["title"],
          author: book_details[0]["author"],
          cover_image_url: book_details[0]["cover_image_url"],
          updated_at: book["updated_at"],
          status: book["status"],
          rating: book["rating"],
        });
        if (book["rating"]) {
          results["ratings"][book["rating"]] =
            (results["ratings"][book["rating"]] || 0) + 1;
        }
        if (book["status"]) {
          results["statuses"][book["status"]] =
            (results["statuses"][book["status"]] || 0) + 1;
        }
      })
    );
    return res
      .status(200)
      .json({ message: "Data Fetched Successfully!", dashboard_data: results });
  } catch (e) {
    return res.status(500).json({ error: "Internal server error", e });
  }
};

export { get_recommendations, get_dashboard };
