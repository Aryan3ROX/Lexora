import { Router } from "express";
import { get_books, get_book } from "./controllers/non_user_controllers.js";
import { registerUser, loginUser, logoutUser } from "./controllers/auth.js";
import { authenticateToken } from "./middlewares/authenticate.js";
import { get_recommendations, get_dashboard } from "./controllers/user_controllers.js";
const router = Router()

router.route("/get-books").post(get_books)

router.route("/get-book").post(get_book)

router.route("/register").post(registerUser)

router.route("/login").post(loginUser)

router.route("/logout").post(logoutUser)

router.route("/get-recommendations").get(authenticateToken, get_recommendations)

router.route("/get-dashboard").get(authenticateToken, get_dashboard)

export default router