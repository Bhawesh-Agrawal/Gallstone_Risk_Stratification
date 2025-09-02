import AboutUs from "./Components/AboutUs/page";
import Footer from "./Components/Footer/page";
import LandingPage from "./Components/LandingPage/page";
import Navbar from "./Components/Navbar/page";
import Process from "./Components/Process/page";
import Quality from "./Components/Quality/page";
import Testimonials from "./Components/Testimonials/page";

export default function Home(){
  return(
    <div>
      <div>
        <LandingPage />
      </div>
      <div>
        <Quality />
      </div>
      <div>
        <AboutUs />
      </div>
      <div>
        <Testimonials />
      </div>
      <div>
        <Process />
      </div>
    </div>
  )
}