"""
Enhanced Renewable Energy Data Extractor with Polygon Support
Version: 3.0 - With Infrastructure Analysis & Restricted Area Detection
Author: Atharva Renewable AI Team
"""

import hashlib
import requests
import json
import random
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
from math import radians, sin, cos, sqrt, atan2


class DataExtractor:
    def __init__(self):
        self.weather_apis = [
            {"name": "Open-Meteo", "type": "open-meteo"},
            {"name": "Weatherbit", "type": "weatherbit", "api_key": "9a1ed9a8bd9d4e52b59889854b904bc6"},
            {"name": "Visual Crossing", "type": "visual-crossing", "api_key": "WEZJL49X6X7C6S6M8TENPDX2G"},
        ]
        self.osm_url = "https://overpass-api.de/api/interpreter"

    def haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in kilometers"""
        R = 6371  # Earth's radius in km
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        return R * c

    # ---------------- MAIN EXTRACTION ---------------- #
    def extract_from_polygon(
        self,
        polygon_coords: List[Tuple[float, float]],
        save_to_file: bool = True,
        output_file: Optional[str] = None,
    ) -> Dict:

        # Compute centroid for weather API call
        centroid_lat = sum([p[0] for p in polygon_coords]) / len(polygon_coords)
        centroid_lon = sum([p[1] for p in polygon_coords]) / len(polygon_coords)

        # Check proximity to restricted areas (NEW - Enhanced)
        restriction_result = self._check_restricted_area_detailed(polygon_coords, centroid_lat, centroid_lon)
        
        if restriction_result["is_restricted"]:
            return {
                "success": False,
                "restricted": True,
                "restriction_details": restriction_result,
                "message": "⚠️ Site is in or near restricted area. Project cannot proceed."
            }

        # Fetch weather data (with fallback)
        weather_data = self._extract_weather_with_fallback(centroid_lat, centroid_lon)

        # Fetch OSM data for polygon
        osm_data = self._extract_osm_data_polygon(polygon_coords)

        # NEW - Get infrastructure analysis
        infra_analysis = self._analyze_infrastructure(centroid_lat, centroid_lon)

        # Compute statistics if available
        weather_stats = (
            self._calculate_statistics(weather_data) if weather_data and "daily" in weather_data else None
        )

        # Generate structured site data for Solar and Wind
        solar_sample, wind_sample = self.generate_site_samples(weather_stats)

        result = {
            "success": True,
            "restricted": False,
            "metadata": {
                "polygon_coords": polygon_coords,
                "centroid": (centroid_lat, centroid_lon),
                "source": weather_data.get("source") if weather_data else "Synthetic",
                "timestamp": datetime.now().isoformat(),
            },
            "solar_site": solar_sample,
            "wind_site": wind_sample,
            "weather_data": weather_data,
            "infrastructure_data": osm_data,
            "infrastructure_analysis": infra_analysis,  # NEW
            "restriction_check": restriction_result,    # NEW
        }

        # Save results if needed
        if save_to_file:
            if not output_file:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"renewable_data_polygon_{timestamp}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            print(f"💾 Data saved to: {output_file}")

        self._print_summary(result)
        return result

    # ---------------- ENHANCED RESTRICTED AREA CHECK ---------------- #
    def _check_restricted_area_detailed(self, polygon_coords: List[Tuple[float, float]], 
                                       centroid_lat: float, centroid_lon: float) -> Dict:
        """Enhanced restricted area check with detailed distance information"""
        
        min_lat = min(p[0] for p in polygon_coords) - 0.1
        max_lat = max(p[0] for p in polygon_coords) + 0.1
        min_lon = min(p[1] for p in polygon_coords) - 0.1
        max_lon = max(p[1] for p in polygon_coords) + 0.1
        
        # Safe distances in km
        SAFE_DISTANCES = {
            "school": 1.0,      # 1 km from schools
            "hospital": 0.5,    # 0.5 km from hospitals
            "airport": 10.0,    # 10 km from airports
            "residential": 0.5, # 0.5 km from residential areas
            "military": 5.0,    # 5 km from military bases
            "heritage": 2.0,    # 2 km from heritage sites
        }
        
        query = f"""
        [out:json][timeout:30];
        (
          node["amenity"="school"]({min_lat},{min_lon},{max_lat},{max_lon});
          node["amenity"="hospital"]({min_lat},{min_lon},{max_lat},{max_lon});
          node["amenity"="kindergarten"]({min_lat},{min_lon},{max_lat},{max_lon});
          node["aeroway"="aerodrome"]({min_lat},{min_lon},{max_lat},{max_lon});
          way["aeroway"="aerodrome"]({min_lat},{min_lon},{max_lat},{max_lon});
          node["landuse"="residential"]({min_lat},{min_lon},{max_lat},{max_lon});
          way["landuse"="residential"]({min_lat},{min_lon},{max_lat},{max_lon});
          node["military"]({min_lat},{min_lon},{max_lat},{max_lon});
          way["military"]({min_lat},{min_lon},{max_lat},{max_lon});
          node["historic"]({min_lat},{min_lon},{max_lat},{max_lon});
        );
        out center;
        """
        
        result = {
            "is_restricted": False,
            "violations": [],
            "nearby_facilities": [],
            "safe_distances_met": True
        }
        
        try:
            resp = requests.post(self.osm_url, data={"data": query}, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            for element in data.get("elements", []):
                tags = element.get("tags", {})
                
                # Get coordinates
                if "center" in element:
                    elem_lat = element["center"]["lat"]
                    elem_lon = element["center"]["lon"]
                elif "lat" in element:
                    elem_lat = element["lat"]
                    elem_lon = element["lon"]
                else:
                    continue
                
                # Calculate distance from centroid
                distance = self.haversine_distance(centroid_lat, centroid_lon, elem_lat, elem_lon)
                
                # Determine facility type
                facility_type = None
                facility_name = tags.get("name", "Unknown")
                
                if tags.get("amenity") == "school" or tags.get("amenity") == "kindergarten":
                    facility_type = "school"
                elif tags.get("amenity") == "hospital":
                    facility_type = "hospital"
                elif tags.get("aeroway") == "aerodrome":
                    facility_type = "airport"
                    facility_name = tags.get("name", "Airport")
                elif tags.get("landuse") == "residential":
                    facility_type = "residential"
                elif tags.get("military"):
                    facility_type = "military"
                elif tags.get("historic"):
                    facility_type = "heritage"
                
                if facility_type:
                    safe_distance = SAFE_DISTANCES.get(facility_type, 1.0)
                    is_violation = distance < safe_distance
                    
                    facility_info = {
                        "type": facility_type,
                        "name": facility_name,
                        "distance_km": round(distance, 2),
                        "safe_distance_km": safe_distance,
                        "is_violation": is_violation
                    }
                    
                    result["nearby_facilities"].append(facility_info)
                    
                    if is_violation:
                        result["is_restricted"] = True
                        result["safe_distances_met"] = False
                        result["violations"].append({
                            "facility": facility_name,
                            "type": facility_type,
                            "distance": round(distance, 2),
                            "required_distance": safe_distance,
                            "shortage": round(safe_distance - distance, 2)
                        })
            
            # Sort facilities by distance
            result["nearby_facilities"].sort(key=lambda x: x["distance_km"])
            
            print(f"✓ Restriction check: {'RESTRICTED' if result['is_restricted'] else 'CLEAR'}")
            print(f"  Found {len(result['nearby_facilities'])} nearby facilities")
            if result["violations"]:
                print(f"  ⚠️ {len(result['violations'])} safety violations detected")
                
        except Exception as e:
            print(f"⚠️ Restriction check failed: {e}")
            result["error"] = str(e)
        
        return result

    # ---------------- INFRASTRUCTURE ANALYSIS ---------------- #
    def _analyze_infrastructure(self, lat: float, lon: float) -> Dict:
        """Analyze nearby infrastructure for project feasibility"""
        
        # Search radius: 50km
        search_radius = 0.45  # ~50km in degrees
        
        query = f"""
        [out:json][timeout:30];
        (
          way["power"="line"]({lat-search_radius},{lon-search_radius},{lat+search_radius},{lon+search_radius});
          node["power"="substation"]({lat-search_radius},{lon-search_radius},{lat+search_radius},{lon+search_radius});
          way["power"="substation"]({lat-search_radius},{lon-search_radius},{lat+search_radius},{lon+search_radius});
          way["highway"~"motorway|trunk|primary"]({lat-search_radius},{lon-search_radius},{lat+search_radius},{lon+search_radius});
          way["railway"="rail"]({lat-search_radius},{lon-search_radius},{lat+search_radius},{lon+search_radius});
          node["amenity"="fuel"]({lat-search_radius},{lon-search_radius},{lat+search_radius},{lon+search_radius});
        );
        out center;
        """
        
        analysis = {
            "power_grid": {"distance_km": None, "nearest": None, "count": 0},
            "substation": {"distance_km": None, "nearest": None, "count": 0},
            "major_road": {"distance_km": None, "nearest": None, "count": 0},
            "railway": {"distance_km": None, "nearest": None, "count": 0},
            "fuel_station": {"distance_km": None, "nearest": None, "count": 0},
            "connectivity_score": 0,
        }
        
        try:
            resp = requests.post(self.osm_url, data={"data": query}, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            for element in data.get("elements", []):
                tags = element.get("tags", {})
                
                # Get coordinates
                if "center" in element:
                    elem_lat = element["center"]["lat"]
                    elem_lon = element["center"]["lon"]
                elif "lat" in element:
                    elem_lat = element["lat"]
                    elem_lon = element["lon"]
                else:
                    continue
                
                distance = self.haversine_distance(lat, lon, elem_lat, elem_lon)
                
                # Categorize infrastructure
                if tags.get("power") == "line":
                    analysis["power_grid"]["count"] += 1
                    if not analysis["power_grid"]["distance_km"] or distance < analysis["power_grid"]["distance_km"]:
                        analysis["power_grid"]["distance_km"] = round(distance, 2)
                        analysis["power_grid"]["nearest"] = tags.get("name", "Power Line")
                
                elif tags.get("power") == "substation":
                    analysis["substation"]["count"] += 1
                    if not analysis["substation"]["distance_km"] or distance < analysis["substation"]["distance_km"]:
                        analysis["substation"]["distance_km"] = round(distance, 2)
                        analysis["substation"]["nearest"] = tags.get("name", "Substation")
                
                elif tags.get("highway") in ["motorway", "trunk", "primary"]:
                    analysis["major_road"]["count"] += 1
                    if not analysis["major_road"]["distance_km"] or distance < analysis["major_road"]["distance_km"]:
                        analysis["major_road"]["distance_km"] = round(distance, 2)
                        analysis["major_road"]["nearest"] = tags.get("name", f"{tags.get('highway', 'Road').title()} Road")
                
                elif tags.get("railway") == "rail":
                    analysis["railway"]["count"] += 1
                    if not analysis["railway"]["distance_km"] or distance < analysis["railway"]["distance_km"]:
                        analysis["railway"]["distance_km"] = round(distance, 2)
                        analysis["railway"]["nearest"] = tags.get("name", "Railway")
                
                elif tags.get("amenity") == "fuel":
                    analysis["fuel_station"]["count"] += 1
                    if not analysis["fuel_station"]["distance_km"] or distance < analysis["fuel_station"]["distance_km"]:
                        analysis["fuel_station"]["distance_km"] = round(distance, 2)
                        analysis["fuel_station"]["nearest"] = tags.get("name", "Fuel Station")
            
            # Calculate connectivity score (0-100)
            score = 0
            if analysis["power_grid"]["distance_km"] and analysis["power_grid"]["distance_km"] < 20:
                score += 30
            elif analysis["power_grid"]["distance_km"] and analysis["power_grid"]["distance_km"] < 50:
                score += 15
            
            if analysis["substation"]["distance_km"] and analysis["substation"]["distance_km"] < 10:
                score += 25
            elif analysis["substation"]["distance_km"] and analysis["substation"]["distance_km"] < 30:
                score += 10
            
            if analysis["major_road"]["distance_km"] and analysis["major_road"]["distance_km"] < 5:
                score += 20
            elif analysis["major_road"]["distance_km"] and analysis["major_road"]["distance_km"] < 15:
                score += 10
            
            if analysis["railway"]["distance_km"] and analysis["railway"]["distance_km"] < 20:
                score += 15
            
            if analysis["fuel_station"]["distance_km"] and analysis["fuel_station"]["distance_km"] < 10:
                score += 10
            
            analysis["connectivity_score"] = score
            
            print(f"✓ Infrastructure analysis complete")
            print(f"  Connectivity Score: {score}/100")
            
        except Exception as e:
            print(f"⚠️ Infrastructure analysis failed: {e}")
            analysis["error"] = str(e)
        
        return analysis

    # ---------------- WEATHER API (unchanged) ---------------- #
    def _extract_weather_with_fallback(self, lat: float, lon: float) -> Dict:
        print("📡 Fetching Weather Data...")
        for api in self.weather_apis:
            try:
                if api["type"] == "open-meteo":
                    data = self._fetch_open_meteo(lat, lon)
                elif api["type"] == "weatherbit":
                    data = self._fetch_weatherbit(lat, lon, api.get("api_key"))
                elif api["type"] == "visual-crossing":
                    data = self._fetch_visual_crossing(lat, lon, api.get("api_key"))
                else:
                    continue
                if data:
                    data["source"] = api["name"]
                    print(f"✓ {api['name']} Success")
                    return data
            except Exception as e:
                print(f"✗ {api['name']} Failed: {e}")
                continue
        print("⚠ All APIs failed — using synthetic fallback data.")
        return self._generate_fake_weather()

    def _fetch_open_meteo(self, latitude: float, longitude: float) -> Dict:
        end_date = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "daily": ",".join([
                "temperature_2m_mean", "precipitation_sum", "snowfall_sum",
                "wind_speed_10m_max", "wind_gusts_10m_max", "relative_humidity_2m_mean",
                "surface_pressure_mean", "cloud_cover_mean", "shortwave_radiation_sum",
                "sunshine_duration",
            ]),
            "timezone": "auto",
        }
        resp = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def _fetch_weatherbit(self, lat, lon, key):
        params = {
            "lat": lat, "lon": lon,
            "start_date": (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
            "end_date": (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
            "key": key,
        }
        r = requests.get("https://api.weatherbit.io/v2.0/history/daily", params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    def _fetch_visual_crossing(self, lat, lon, key):
        url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lon}/last30days"
        params = {"key": key, "unitGroup": "metric"}
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    def _generate_fake_weather(self) -> Dict:
        days = [(datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(365)]
        return {
            "daily": {
                "time": days[::-1],
                "temperature_2m_mean": [random.uniform(10, 35) for _ in days],
                "precipitation_sum": [random.uniform(0, 10) for _ in days],
                "snowfall_sum": [random.uniform(0, 2) for _ in days],
                "wind_speed_10m_max": [random.uniform(2, 8) for _ in days],
                "wind_gusts_10m_max": [random.uniform(5, 12) for _ in days],
                "relative_humidity_2m_mean": [random.uniform(60, 90) for _ in days],
                "surface_pressure_mean": [random.uniform(990, 1020) for _ in days],
                "cloud_cover_mean": [random.uniform(40, 90) for _ in days],
                "shortwave_radiation_sum": [random.uniform(2, 6) * 1000 for _ in days],
                "sunshine_duration": [random.uniform(2, 8) for _ in days],
            },
            "source": "Synthetic Random",
        }

    def _calculate_statistics(self, weather_data: Dict) -> Dict:
        from collections import defaultdict
        daily = weather_data.get("daily", {})
        times = daily.get("time", [])
        quarterly = defaultdict(lambda: defaultdict(list))
        for i, t in enumerate(times):
            try:
                date = datetime.fromisoformat(t)
            except Exception:
                continue
            q = f"Q{(date.month - 1)//3 + 1}"
            for k, vals in daily.items():
                if k == "time" or i >= len(vals):
                    continue
                quarterly[k][q].append(float(vals[i]))
                quarterly[k]["yearly"].append(float(vals[i]))
        def avg(lst): return sum(lst)/len(lst) if lst else 0
        stats = {}
        for k, qs in quarterly.items():
            stats[k] = {"yearly": {"avg": avg(qs["yearly"]), "sum": sum(qs["yearly"])}, "quarterly": {}}
            for q, lst in qs.items():
                if q != "yearly":
                    stats[k]["quarterly"][q] = {"avg": avg(lst), "sum": sum(lst)}
        return stats

    def _extract_osm_data_polygon(self, polygon_coords: List[Tuple[float, float]]) -> Dict:
        print("🗺 Fetching OSM Infrastructure Data within polygon...")
        poly_str = " ".join(f"{lat} {lon}" for lat, lon in polygon_coords)
        query = f"""
        [out:json][timeout:30];
        (
          way(poly:"{poly_str}")["power"];
          node(poly:"{poly_str}")["power"];
          way(poly:"{poly_str}")["highway"];
          way(poly:"{poly_str}")["railway"];
          way(poly:"{poly_str}")["landuse"~"industrial|commercial"];
        );
        out geom;
        """
        try:
            resp = requests.post(self.osm_url, data={"data": query}, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"✗ Failed: {e}")
            return {}

    def generate_site_samples(self, stats: Optional[Dict], polygon_coords: Optional[list] = None) -> tuple:
        """
        Generate solar and wind site samples.
        Even with synthetic data, parameters vary per site based on polygon coordinates.
        """
        # Use polygon hash to seed random for site-specific variation
        if polygon_coords:
            polygon_seed = int(hashlib.sha256(str(polygon_coords).encode()).hexdigest(), 16) % 10**8
        else:
            polygon_seed = random.randint(0, 10**8)
        rnd = random.Random(polygon_seed)

        # Helper functions
        def safe(param, quarter, fallback_range):
            try:
                val = stats[param]["quarterly"][quarter]["avg"]
                return round(val, 2)
            except Exception:
                return round(rnd.uniform(fallback_range[0], fallback_range[1]), 2)

        def safe_year(param, fallback_range):
            try:
                val = stats[param]["yearly"]["avg"]
                return round(val, 2)
            except Exception:
                return round(rnd.uniform(fallback_range[0], fallback_range[1]), 2)

        # ---------------- SOLAR SITE ---------------- #
        solar_site = {
            "GHI (kWh/m²/day)": safe_year("shortwave_radiation_sum", [3, 6]),
            "DNI (kWh/m²/day)": rnd.uniform(2.5, 5),
            "DHI (% of GHI)": rnd.uniform(20, 40),
            "Snowfall (mm/year)": safe_year("snowfall_sum", [0, 200]),
        }
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            solar_site[f"{q}-Cloud cover"] = safe("cloud_cover_mean", q, [60, 85])
            solar_site[f"{q}-Sunshine duration"] = safe("sunshine_duration", q, [3, 6])
            solar_site[f"{q}-Ambient temperature"] = safe("temperature_2m_mean", q, [10, 25])
            solar_site[f"{q}-Relative humidity"] = safe("relative_humidity_2m_mean", q, [70, 90])
            solar_site[f"{q}-Precipitation"] = safe("precipitation_sum", q, [400, 900])
        solar_site.update({
            "YearlyCloud cover": safe_year("cloud_cover_mean", [70, 85]),
            "Sunshine duration": safe_year("sunshine_duration", [3, 6]),
            "Ambient temperature": safe_year("temperature_2m_mean", [12, 25]),
            "Relative humidity": safe_year("relative_humidity_2m_mean", [75, 90]),
            "Precipitation": safe_year("precipitation_sum", [2000, 3000]),
        })

        # ---------------- WIND SITE ---------------- #
        wind_site = {
            "Slope": rnd.uniform(5, 20),
            "Elevation": rnd.uniform(500, 1500),
            "TurbulenceIntensity": rnd.uniform(10, 20),
        }
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            wind_site[f"{q}-AirTemperature"] = safe("temperature_2m_mean", q, [10, 20])
            wind_site[f"{q}-RelativeHumidity"] = safe("relative_humidity_2m_mean", q, [75, 90])
            wind_site[f"{q}-Precipitation"] = safe("precipitation_sum", q, [400, 800])
            wind_site[f"{q}-WindSpeed"] = safe("wind_speed_10m_max", q, [3, 9])
            wind_site[f"{q}-WindGustSpeed"] = safe("wind_gusts_10m_max", q, [30, 60])
            wind_site[f"{q}-AirPressure"] = safe("surface_pressure_mean", q, [990, 1025])
        wind_site.update({
            "Yearly-AirTemperature": safe_year("temperature_2m_mean", [12, 18]),
            "Yearly-RelativeHumidity": safe_year("relative_humidity_2m_mean", [75, 90]),
            "Yearly-Precipitation": safe_year("precipitation_sum", [2000, 3000]),
            "Yearly-WindSpeed": safe_year("wind_speed_10m_max", [4, 8]),
            "Yearly-WindGustSpeed": safe_year("wind_gusts_10m_max", [35, 55]),
            "Yearly-AirPressure": safe_year("surface_pressure_mean", [995, 1015]),
        })

        return solar_site, wind_site

    def _print_summary(self, result: Dict):
        if not result.get("success"):
            return
        print("\n📈 DATA SUMMARY")
        print(f"Polygon Centroid: {result['metadata']['centroid']}")
        print(f"OSM Elements: {len(result.get('infrastructure_data', {}).get('elements', []))}")
        print(f"Weather Source: {result['metadata']['source']}")
        print(f"Connectivity Score: {result.get('infrastructure_analysis', {}).get('connectivity_score', 'N/A')}")
        print("=" * 50)

    def export_data(self, polygon_coords: List[Tuple[float, float]]) -> Dict:
        """Call this function from another Python file to get extracted data"""
        return self.extract_from_polygon(polygon_coords, save_to_file=False)


if __name__ == "__main__":
    from solar import analyze_solar_site
    from wind import analyze_wind_site
    
    extractor = DataExtractor()
    polygon = [(18.522, 73.854), (18.522, 73.859), (18.518, 73.859), (18.518, 73.854)]
    data = extractor.extract_from_polygon(polygon)
    
    if data.get("success"):
        print("\n🌞 Solar Site Sample:")
        print(json.dumps(data["solar_site"], indent=2))
        print("\n🌬 Wind Site Sample:")
        print(json.dumps(data["wind_site"], indent=2))
        
        solar_label, solar_score, solar_suggestions = analyze_solar_site(data["solar_site"])
        wind_label, wind_score, wind_suggestions = analyze_wind_site(data["wind_site"])
    else:
        print(f"\n❌ {data.get('message', 'Analysis failed')}")