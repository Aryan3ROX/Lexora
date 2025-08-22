CREATE TABLE books (                                                                                              
  id bigint PRIMARY KEY,
  title text,
  author text,
  pub_date date,
  genres jsonb,
  summary text
);
\copy books (id,title,author,pub_date,genres,summary) FROM 'backend/ai/booksummaries/book_data_clean.csv' WITH (FORMAT csv, HEADER true, FORCE_NULL(pub_date));