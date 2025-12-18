import pool from "../db_connection.js";

const get_books = async (req, res) => {
  const { title, author, year_start, year_end, genres } = req.body;
  try {
    let queryStr =
      "SELECT id, title, author, pub_date, genres, cover_image_url FROM books WHERE 1=1";
    const queryParams = [];
    let paramIdx = 1;

    if (title) {
      queryStr += ` AND title ILIKE $${paramIdx}`;
      queryParams.push(`%${title}%`);
      paramIdx++;
    }

    if (author) {
      queryStr += ` AND author ILIKE $${paramIdx}`;
      queryParams.push(`%${author}%`);
      paramIdx++;
    }

    if (year_start) {
      queryStr += ` AND pub_date >= $${paramIdx}`;
      queryParams.push(`${year_start}-01-01`);
      paramIdx++;
    }

    if (year_end) {
      queryStr += ` AND pub_date <= $${paramIdx}`;
      queryParams.push(`${year_end}-12-31`);
      paramIdx++;
    }

    if (genres && Array.isArray(genres) && genres.length > 0) {
      queryStr += ` AND (`;
      genres.forEach((genre, i) => {
        if (i > 0) queryStr += " OR ";
        queryStr += `genres @> $${paramIdx + i}::jsonb`;
        queryParams.push(JSON.stringify([genre]));
      });
      queryStr += `)`;
      paramIdx += genres.length;
    }

    const results = await pool.query(queryStr, queryParams);

    return res
      .status(200)
      .json({ message: "Fetched Books Successfully!", books: results.rows });
  } catch (e) {
    console.error("Database error:", e);
    res.json({ error: "Database Query Error!", e });
  }
};

const get_book = async (req, res) => {
  const { book_id } = req.body;
  try {
    const result = await pool.query(
      "SELECT id, title, author, pub_date, genres, cover_image_url, summary FROM books WHERE id = $1",
      [book_id]
    );

    return res
      .status(200)
      .json({ message: "Fetched Book!", book_data: result.rows });
  } catch (e) {
    res.json({ error: "Database Query Error!", e });
  }
};

export { get_books, get_book };
