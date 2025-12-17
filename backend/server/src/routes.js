import { Router } from "express";
import { get_books } from "./controllers/non_user_controllers.js";

const router = Router()

router.route("/get-books").post(get_books)

export default router