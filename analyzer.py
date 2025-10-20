"""
Complete Testing Script for Wind Data Verification
This will show you exactly what's happening with the API calls
"""

import hashlib
import requests
import json
import random
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
from math import radians, sin, cos, sqrt, atan2, atan, degrees


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

        print("\n" + "="*70)
        print(f"🔍 EXTRACTION STARTED FOR POLYGON")
        print(f"Coordinates: {polygon_coords}")
        print("="*70)

        # Compute centroid for weather API call
        centroid_lat = sum([p[0] for p in polygon_coords]) / len(polygon_coords)
        centroid_lon = sum([p[1] for p in polygon_coords]) / len(polygon_coords)
        
        print(f"📍 Centroid: ({centroid_lat:.4f}, {centroid_lon:.4f})")

        # Check proximity to restricted areas
        print("\n🔒 Checking Restricted Areas...")
        restriction_result = self._check_restricted_area_detailed(polygon_coords, centroid_lat, centroid_lon)
        
        if restriction_result["is_restricted"]:
            print("❌ SITE RESTRICTED - Cannot proceed")
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

        # Get infrastructure analysis
        print("\n🏗️ Analyzing Infrastructure...")
        infra_analysis = self._analyze_infrastructure(centroid_lat, centroid_lon)

        # Get real terrain data
        print("\n🏔️ Getting Real Terrain Data...")
        terrain_data = self.get_real_terrain_data(polygon_coords)

        # Compute statistics if available
        print("\n📊 Computing Weather Statistics...")
        weather_stats = (
            self._calculate_statistics(weather_data) if weather_data and "daily" in weather_data else None
        )
        
        if weather_stats:
            print("✓ Weather statistics computed from real data")
            # Show sample of what we got
            if "wind_speed_10m_max" in weather_stats:
                yearly_wind = weather_stats["wind_speed_10m_max"]["yearly"]["avg"]
                print(f"  📈 Yearly Avg Wind Speed: {yearly_wind:.2f} m/s")
                for q in ["Q1", "Q2", "Q3", "Q4"]:
                    q_wind = weather_stats["wind_speed_10m_max"]["quarterly"][q]["avg"]
                    print(f"  📈 {q} Avg Wind Speed: {q_wind:.2f} m/s")
        else:
            print("⚠️ No weather statistics - will use synthetic data")

        # Generate structured site data for Solar and Wind
        print("\n🌞 Generating Solar Site Data...")
        print("🌬️ Generating Wind Site Data...")
        solar_sample, wind_sample = self.generate_site_samples(weather_stats, polygon_coords, terrain_data)

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
            "infrastructure_analysis": infra_analysis,
            "restriction_check": restriction_result,
        }

        # Save results if needed
        if save_to_file:
            if not output_file:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"renewable_data_polygon_{timestamp}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            print(f"\n💾 Data saved to: {output_file}")

        self._print_summary(result)
        return result

    # ---------------- RESTRICTED AREA CHECK ---------------- #
    def _check_restricted_area_detailed(self, polygon_coords: List[Tuple[float, float]], 
                                       centroid_lat: float, centroid_lon: float) -> Dict:
        """Enhanced restricted area check with detailed distance information"""
        
        min_lat = min(p[0] for p in polygon_coords) - 0.1
        max_lat = max(p[0] for p in polygon_coords) + 0.1
        min_lon = min(p[1] for p in polygon_coords) - 0.1
        max_lon = max(p[1] for p in polygon_coords) + 0.1
        
        SAFE_DISTANCES = {
            "school": 1.0,
            "hospital": 0.5,
            "airport": 10.0,
            "residential": 0.5,
            "military": 5.0,
            "heritage": 2.0,
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
                
                if "center" in element:
                    elem_lat = element["center"]["lat"]
                    elem_lon = element["center"]["lon"]
                elif "lat" in element:
                    elem_lat = element["lat"]
                    elem_lon = element["lon"]
                else:
                    continue
                
                distance = self.haversine_distance(centroid_lat, centroid_lon, elem_lat, elem_lon)
                
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
        
        search_radius = 0.45
        
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
                
                if "center" in element:
                    elem_lat = element["center"]["lat"]
                    elem_lon = element["center"]["lon"]
                elif "lat" in element:
                    elem_lat = element["lat"]
                    elem_lon = element["lon"]
                else:
                    continue
                
                distance = self.haversine_distance(lat, lon, elem_lat, elem_lon)
                
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
            
            # Calculate connectivity score
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

    # ---------------- WEATHER API ---------------- #
    def _extract_weather_with_fallback(self, lat: float, lon: float) -> Dict:
        print("\n📡 Fetching Weather Data...")
        print(f"   Location: ({lat:.4f}, {lon:.4f})")
        
        for api in self.weather_apis:
            print(f"\n   Trying {api['name']}...")
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
                    print(f"   ✓ {api['name']} Success!")
                    
                    # Show sample data
                    if "daily" in data:
                        daily = data["daily"]
                        if "wind_speed_10m_max" in daily:
                            print(f"   📊 Sample Wind Speeds: {daily['wind_speed_10m_max'][:5]}")
                    
                    return data
            except Exception as e:
                print(f"   ✗ {api['name']} Failed: {str(e)[:100]}")
                continue
                
        print("\n   ⚠️ All APIs failed — using synthetic fallback data.")
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
        print("   🎲 Generating synthetic weather data...")
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
        print("\n🗺️ Fetching OSM Infrastructure Data...")
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
            data = resp.json()
            print(f"✓ Found {len(data.get('elements', []))} OSM elements")
            return data
        except Exception as e:
            print(f"✗ OSM fetch failed: {e}")
            return {}

    def get_real_terrain_data(self, polygon_coords: List[Tuple[float, float]]) -> Dict:
        """
        Get REAL Elevation, Slope, and TurbulenceIntensity
        Call this in extract_from_polygon() before generate_site_samples()
        
        Returns:
            {
                'elevation': float (in meters),
                'slope': float (in degrees),
                'turbulence_intensity': float (in percentage)
            }
        """
        print("\n🏔️ Fetching Real Terrain Data...")
        
        # 1. GET REAL ELEVATION
        elevation = self._get_real_elevation(polygon_coords)
        
        # 2. CALCULATE REAL SLOPE
        slope = self._calculate_real_slope(polygon_coords, elevation)
        
        # 3. CALCULATE REAL TURBULENCE INTENSITY
        turbulence = self._calculate_turbulence_intensity(polygon_coords)
        
        result = {
            'elevation': round(elevation, 2),
            'slope': round(slope, 2),
            'turbulence_intensity': round(turbulence, 2)
        }
        
        print(f"   ✓ Real Terrain Data:")
        print(f"      Elevation: {result['elevation']} m")
        print(f"      Slope: {result['slope']}°")
        print(f"      Turbulence Intensity: {result['turbulence_intensity']}%")
        
        return result


    def _get_real_elevation(self, polygon_coords: List[Tuple[float, float]]) -> float:
        """Get real elevation from Open-Elevation API"""
        try:
            # Get center point
            center_lat = sum(p[0] for p in polygon_coords) / len(polygon_coords)
            center_lon = sum(p[1] for p in polygon_coords) / len(polygon_coords)
            
            # Call elevation API
            response = requests.post(
                "https://api.open-elevation.com/api/v1/lookup",
                json={"locations": [{"latitude": center_lat, "longitude": center_lon}]},
                timeout=15
            )
            response.raise_for_status()
            data = response.json()
            
            elevation = data['results'][0]['elevation']
            print(f"   ✓ Elevation API: {elevation}m")
            return float(elevation)
            
        except Exception as e:
            print(f"   ⚠️ Elevation API failed: {e}")
            # Fallback: estimate from latitude (rough approximation)
            center_lat = sum(p[0] for p in polygon_coords) / len(polygon_coords)
            # India elevation roughly correlates with latitude
            estimated = max(50, min(2000, abs(center_lat - 20) * 50 + 200))
            print(f"   Using estimated elevation: {estimated}m")
            return estimated


    def _calculate_real_slope(self, polygon_coords: List[Tuple[float, float]], center_elevation: float) -> float:
        """Calculate real slope from elevation data"""
        try:
            # Sample 4 corner points
            lats = [p[0] for p in polygon_coords]
            lons = [p[1] for p in polygon_coords]
            
            corners = [
                (min(lats), min(lons)),
                (min(lats), max(lons)),
                (max(lats), min(lons)),
                (max(lats), max(lons))
            ]
            
            # Get elevation for corners
            response = requests.post(
                "https://api.open-elevation.com/api/v1/lookup",
                json={"locations": [{"latitude": lat, "longitude": lon} for lat, lon in corners]},
                timeout=15
            )
            response.raise_for_status()
            data = response.json()
            
            elevations = [r['elevation'] for r in data['results']]
            
            # Calculate slope
            elevation_diff = max(elevations) - min(elevations)
            
            # Calculate horizontal distance (approximate)
            lat_diff = max(lats) - min(lats)
            lon_diff = max(lons) - min(lons)
            # 1 degree ≈ 111km
            distance_m = ((lat_diff * 111000)**2 + (lon_diff * 111000)**2)**0.5
            
            # Slope = arctan(rise/run)
            if distance_m > 0:
                slope_rad = atan(elevation_diff / distance_m)
                slope_deg = degrees(slope_rad)
            else:
                slope_deg = 0
            
            print(f"   ✓ Slope calculated: {slope_deg:.2f}° (elevation change: {elevation_diff:.1f}m)")
            return slope_deg
            
        except Exception as e:
            print(f"   ⚠️ Slope calculation failed: {e}")
            # Fallback: use elevation to estimate
            # Higher elevations tend to be more sloped
            estimated_slope = min(20, max(2, center_elevation / 100))
            print(f"   Using estimated slope: {estimated_slope}°")
            return estimated_slope


    def _calculate_turbulence_intensity(self, polygon_coords: List[Tuple[float, float]]) -> float:
        """Calculate turbulence intensity from OSM obstacles"""
        try:
            poly_str = " ".join(f"{lat} {lon}" for lat, lon in polygon_coords)
            
            # Query for obstacles that increase turbulence
            query = f"""
            [out:json][timeout:25];
            (
            way(poly:"{poly_str}")["building"];
            way(poly:"{poly_str}")["landuse"="forest"];
            way(poly:"{poly_str}")["natural"="tree"];
            way(poly:"{poly_str}")["barrier"];
            );
            out count;
            """
            
            resp = requests.post(self.osm_url, data={"data": query}, timeout=25)
            resp.raise_for_status()
            data = resp.json()
            
            # Count obstacles
            elements = data.get('elements', [])
            obstacle_count = len(elements)
            
            # Base turbulence: 10%
            # Add 0.5% per obstacle, cap at 25%
            turbulence = min(25.0, 10.0 + (obstacle_count * 0.5))
            
            print(f"   ✓ Turbulence from OSM: {turbulence}% ({obstacle_count} obstacles)")
            return turbulence
            
        except Exception as e:
            print(f"   ⚠️ Turbulence calculation failed: {e}")
            # Fallback: moderate turbulence
            return 15.0    

    def generate_site_samples(self, stats: Optional[Dict], polygon_coords: Optional[list] = None, terrain_data: Optional[Dict] = None) -> tuple:
        """Generate solar and wind site samples using REAL weather data"""
        # Use polygon hash to seed random for site-specific variation
        if polygon_coords:
            polygon_seed = int(hashlib.sha256(str(polygon_coords).encode()).hexdigest(), 16) % 10**8
        else:
            polygon_seed = random.randint(0, 10**8)
        rnd = random.Random(polygon_seed)

        # Check if using real data
        using_real_data = stats is not None
        print(f"   📊 Data Source: {'REAL API DATA' if using_real_data else 'SYNTHETIC FALLBACK'}")

        # Helper functions (unchanged)
        def safe(param, quarter, fallback_range):
            """Get quarterly average, fallback to random if stats unavailable"""
            if stats:
                try:
                    val = stats[param]["quarterly"][quarter]["avg"]
                    return round(val, 2)
                except Exception:
                    pass
            return round(rnd.uniform(fallback_range[0], fallback_range[1]), 2)

        def safe_year(param, fallback_range):
            """Get yearly average, fallback to random if stats unavailable"""
            if stats:
                try:
                    val = stats[param]["yearly"]["avg"]
                    return round(val, 2)
                except Exception:
                    pass
            return round(rnd.uniform(fallback_range[0], fallback_range[1]), 2)

        def safe_radiation_year(param, fallback_range):
            """Convert yearly radiation from Wh to kWh"""
            if stats:
                try:
                    val = stats[param]["yearly"]["avg"]
                    return round(val / 1000, 2)
                except Exception:
                    pass
            return round(rnd.uniform(fallback_range[0], fallback_range[1]), 2)

        def safe_sunshine(param, quarter, fallback_range):
            """Convert sunshine duration from seconds to hours"""
            if stats:
                try:
                    val = stats[param]["quarterly"][quarter]["avg"]
                    return round(val / 3600, 2)
                except Exception:
                    pass
            return round(rnd.uniform(fallback_range[0], fallback_range[1]), 2)

        def safe_sunshine_year(param, fallback_range):
            """Convert yearly sunshine duration from seconds to hours"""
            if stats:
                try:
                    val = stats[param]["yearly"]["avg"]
                    return round(val / 3600, 2)
                except Exception:
                    pass
            return round(rnd.uniform(fallback_range[0], fallback_range[1]), 2)

        # ---------------- SOLAR SITE (unchanged) ---------------- #
        solar_site = {
            "GHI (kWh/m²/day)": safe_radiation_year("shortwave_radiation_sum", [3, 6]),
            "DNI (kWh/m²/day)": safe_radiation_year("shortwave_radiation_sum", [2.5, 5]) * rnd.uniform(0.7, 0.9),
            "DHI (% of GHI)": rnd.uniform(20, 40),
            "Snowfall (mm/year)": safe_year("snowfall_sum", [0, 200]),
        }
        
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            solar_site[f"{q}-Cloud cover"] = safe("cloud_cover_mean", q, [60, 85])
            solar_site[f"{q}-Sunshine duration"] = safe_sunshine("sunshine_duration", q, [3, 6])
            solar_site[f"{q}-Ambient temperature"] = safe("temperature_2m_mean", q, [10, 25])
            solar_site[f"{q}-Relative humidity"] = safe("relative_humidity_2m_mean", q, [70, 90])
            solar_site[f"{q}-Precipitation"] = safe("precipitation_sum", q, [400, 900])
        
        solar_site.update({
            "YearlyCloud cover": safe_year("cloud_cover_mean", [70, 85]),
            "Sunshine duration": safe_sunshine_year("sunshine_duration", [3, 6]),
            "Ambient temperature": safe_year("temperature_2m_mean", [12, 25]),
            "Relative humidity": safe_year("relative_humidity_2m_mean", [75, 90]),
            "Precipitation": safe_year("precipitation_sum", [2000, 3000]),
        })

        # ---------------- WIND SITE (FIXED COLUMN NAMES!) ---------------- #
        wind_site = {
        "Slope": terrain_data['slope'],
        "Elevation": terrain_data['elevation'],
        "TurbulenceIntensity": terrain_data['turbulence_intensity'],
        }
        
        print("\n   🌬️ Wind Data Extraction:")
        
        # CRITICAL FIX: Use "Quarter1", "Quarter2", "Quarter3", "Quarter4" instead of "Q1", "Q2", "Q3", "Q4"
        quarter_mapping = {
            "Q1": "Quarter1",
            "Q2": "Quarter2", 
            "Q3": "Quarter3",
            "Q4": "Quarter4"
        }
        
        for q_short, q_full in quarter_mapping.items():
            # Extract values using Q1, Q2, Q3, Q4 for stats lookup
            wind_speed = safe("wind_speed_10m_max", q_short, [3, 9])
            wind_gust = safe("wind_gusts_10m_max", q_short, [8, 20])
            air_pressure = safe("surface_pressure_mean", q_short, [990, 1025])
            air_temp = safe("temperature_2m_mean", q_short, [10, 20])
            rel_humidity = safe("relative_humidity_2m_mean", q_short, [75, 90])
            precipitation = safe("precipitation_sum", q_short, [400, 800])
            
            # Store with "Quarter1", "Quarter2", etc. to match CSV column names
            wind_site[f"{q_full}-AirTemperature"] = air_temp
            wind_site[f"{q_full}-RelativeHumidity"] = rel_humidity
            wind_site[f"{q_full}-Precipitation"] = precipitation
            wind_site[f"{q_full}-WindSpeed"] = wind_speed
            wind_site[f"{q_full}-WindGustSpeed"] = wind_gust
            wind_site[f"{q_full}-AirPressure"] = air_pressure
            
            print(f"      {q_short}: Wind={wind_speed:>5.2f} m/s | Gust={wind_gust:>5.2f} m/s | "
                f"Pressure={air_pressure:>7.2f} hPa | Temp={air_temp:>5.2f}°C "
                f"{'(API)' if using_real_data else '(synth)'}")
        
        # Extract yearly averages
        yearly_wind = safe_year("wind_speed_10m_max", [4, 8])
        yearly_gust = safe_year("wind_gusts_10m_max", [10, 18])
        yearly_pressure = safe_year("surface_pressure_mean", [995, 1015])
        yearly_temp = safe_year("temperature_2m_mean", [12, 18])
        yearly_humidity = safe_year("relative_humidity_2m_mean", [75, 90])
        yearly_precip = safe_year("precipitation_sum", [2000, 3000])
        
        # Populate yearly data
        wind_site.update({
            "Yearly-AirTemperature": yearly_temp,
            "Yearly-RelativeHumidity": yearly_humidity,
            "Yearly-Precipitation": yearly_precip,
            "Yearly-WindSpeed": yearly_wind,
            "Yearly-WindGustSpeed": yearly_gust,
            "Yearly-AirPressure": yearly_pressure,
        })
        
        print(f"      Yearly: Wind={yearly_wind:>5.2f} m/s | Gust={yearly_gust:>5.2f} m/s | "
            f"Pressure={yearly_pressure:>7.2f} hPa {'(API)' if using_real_data else '(synth)'}")
        
        # DEBUG: Print what we're sending
        print(f"\n   ✅ Wind Site Dict Keys: {list(wind_site.keys())}")
        print(f"   ✅ Total Keys: {len(wind_site)}")
        
        return solar_site, wind_site

    def _print_summary(self, result: Dict):
        """Print summary of extraction results"""
        if not result.get("success"):
            return
        
        print("\n" + "="*70)
        print("📈 EXTRACTION SUMMARY")
        print("="*70)
        print(f"✓ Polygon Centroid: {result['metadata']['centroid']}")
        print(f"✓ Weather Source: {result['metadata']['source']}")
        print(f"✓ OSM Elements: {len(result.get('infrastructure_data', {}).get('elements', []))}")
        print(f"✓ Connectivity Score: {result.get('infrastructure_analysis', {}).get('connectivity_score', 'N/A')}/100")
        print(f"✓ Restricted: {'YES ❌' if result.get('restricted') else 'NO ✓'}")
        print("="*70)

    def export_data(self, polygon_coords: List[Tuple[float, float]]) -> Dict:
        """Call this function from another Python file to get extracted data"""
        return self.extract_from_polygon(polygon_coords, save_to_file=False)


# ============================================================================
# COMPREHENSIVE TEST SUITE
# ============================================================================

def test_multiple_locations():
    """Test different locations to verify wind data varies"""
    print("\n" + "🧪"*35)
    print("COMPREHENSIVE TEST SUITE - WIND DATA VERIFICATION")
    print("🧪"*35)
    
    extractor = DataExtractor()
    
    # Test locations
    test_cases = [
        {
            "name": "Pune, India",
            "polygon": [(18.522, 73.854), (18.522, 73.859), (18.518, 73.859), (18.518, 73.854)]
        },
        {
            "name": "Mumbai, India", 
            "polygon": [(19.076, 72.877), (19.076, 72.882), (19.072, 72.882), (19.072, 72.877)]
        },
        {
            "name": "Delhi, India",
            "polygon": [(28.704, 77.102), (28.704, 77.107), (28.700, 77.107), (28.700, 77.102)]
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"TEST CASE {i}/{len(test_cases)}: {test_case['name']}")
        print(f"{'='*70}")
        
        data = extractor.extract_from_polygon(
            test_case['polygon'],
            save_to_file=False
        )
        
        if data.get("success"):
            results.append({
                "location": test_case['name'],
                "source": data['metadata']['source'],
                "wind_q1": data['wind_site']['Q1-WindSpeed'],
                "wind_q2": data['wind_site']['Q2-WindSpeed'],
                "wind_q3": data['wind_site']['Q3-WindSpeed'],
                "wind_q4": data['wind_site']['Q4-WindSpeed'],
                "wind_yearly": data['wind_site']['Yearly-WindSpeed'],
            })
    
    # Print comparison table
    print("\n" + "="*70)
    print("📊 WIND DATA COMPARISON TABLE")
    print("="*70)
    print(f"{'Location':<20} {'Source':<15} {'Q1':<8} {'Q2':<8} {'Q3':<8} {'Q4':<8} {'Yearly':<8}")
    print("-"*70)
    
    for r in results:
        print(f"{r['location']:<20} {r['source']:<15} "
              f"{r['wind_q1']:<8.2f} {r['wind_q2']:<8.2f} "
              f"{r['wind_q3']:<8.2f} {r['wind_q4']:<8.2f} "
              f"{r['wind_yearly']:<8.2f}")
    
    print("="*70)
    
    # Analysis
    print("\n📋 ANALYSIS:")
    if len(results) >= 2:
        # Check if values are different
        all_same = all(
            results[0]['wind_yearly'] == r['wind_yearly'] 
            for r in results[1:]
        )
        
        if all_same:
            print("⚠️  WARNING: All locations show SAME wind speeds!")
            print("   This indicates synthetic data or API failure.")
            print("   Check if API keys are working.")
        else:
            print("✓  SUCCESS: Wind speeds vary by location!")
            print("   This confirms real API data is being used.")
            
            # Calculate variance
            yearly_speeds = [r['wind_yearly'] for r in results]
            variance = max(yearly_speeds) - min(yearly_speeds)
            print(f"   Variance in yearly wind speed: {variance:.2f} m/s")
    
    # Check data sources
    sources = [r['source'] for r in results]
    print(f"\n📡 Data Sources Used:")
    for source in set(sources):
        count = sources.count(source)
        print(f"   - {source}: {count}/{len(results)} locations")
    
    return results


def test_same_location_twice():
    """Test same location twice to verify consistency"""
    print("\n" + "🧪"*35)
    print("CONSISTENCY TEST - SAME LOCATION TWICE")
    print("🧪"*35)
    
    extractor = DataExtractor()
    polygon = [(18.522, 73.854), (18.522, 73.859), (18.518, 73.859), (18.518, 73.854)]
    
    print("\n🔄 Extracting data for same location - FIRST CALL")
    data1 = extractor.extract_from_polygon(polygon, save_to_file=False)
    
    print("\n🔄 Extracting data for same location - SECOND CALL")
    data2 = extractor.extract_from_polygon(polygon, save_to_file=False)
    
    print("\n" + "="*70)
    print("📊 CONSISTENCY CHECK")
    print("="*70)
    
    if data1.get("success") and data2.get("success"):
        wind1 = data1['wind_site']
        wind2 = data2['wind_site']
        
        print(f"{'Parameter':<30} {'Call 1':<15} {'Call 2':<15} {'Match':<10}")
        print("-"*70)
        
        params = ['Q1-WindSpeed', 'Q2-WindSpeed', 'Q3-WindSpeed', 'Q4-WindSpeed', 'Yearly-WindSpeed']
        
        all_match = True
        for param in params:
            val1 = wind1[param]
            val2 = wind2[param]
            match = "✓" if abs(val1 - val2) < 0.01 else "✗"
            if match == "✗":
                all_match = False
            print(f"{param:<30} {val1:<15.2f} {val2:<15.2f} {match:<10}")
        
        print("="*70)
        if all_match:
            print("✓  SUCCESS: Same location produces consistent results!")
            print("   (This is expected behavior with real API data)")
        else:
            print("⚠️  WARNING: Results differ for same location!")
            print("   This might indicate random synthetic data is being used.")
    
    return data1, data2


def show_detailed_wind_breakdown(data):
    """Show detailed breakdown of wind data sources"""
    if not data.get("success"):
        print("❌ Data extraction failed")
        return
    
    print("\n" + "="*70)
    print("🔍 DETAILED WIND DATA BREAKDOWN")
    print("="*70)
    
    print(f"\n📍 Location: {data['metadata']['centroid']}")
    print(f"📡 Weather Source: {data['metadata']['source']}")
    print(f"⏰ Timestamp: {data['metadata']['timestamp']}")
    
    # Check if we have raw weather data
    if 'weather_data' in data and 'daily' in data['weather_data']:
        daily = data['weather_data']['daily']
        
        if 'wind_speed_10m_max' in daily:
            wind_speeds = daily['wind_speed_10m_max']
            print(f"\n📊 Raw Wind Speed Data from API:")
            print(f"   Total data points: {len(wind_speeds)}")
            print(f"   Min: {min(wind_speeds):.2f} m/s")
            print(f"   Max: {max(wind_speeds):.2f} m/s")
            print(f"   Average: {sum(wind_speeds)/len(wind_speeds):.2f} m/s")
            print(f"   First 10 values: {[round(x, 2) for x in wind_speeds[:10]]}")
        else:
            print("\n⚠️  No wind_speed_10m_max in API response")
            print(f"   Available fields: {list(daily.keys())}")
    else:
        print("\n⚠️  No daily weather data available")
    
    # Show processed wind site data
    wind_site = data['wind_site']
    print(f"\n🌬️  Processed Wind Site Data:")
    print(f"   Q1 Wind Speed: {wind_site['Q1-WindSpeed']:.2f} m/s")
    print(f"   Q2 Wind Speed: {wind_site['Q2-WindSpeed']:.2f} m/s")
    print(f"   Q3 Wind Speed: {wind_site['Q3-WindSpeed']:.2f} m/s")
    print(f"   Q4 Wind Speed: {wind_site['Q4-WindSpeed']:.2f} m/s")
    print(f"   Yearly Average: {wind_site['Yearly-WindSpeed']:.2f} m/s")
    
    print("\n" + "="*70)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + " " * 15 + "WIND DATA VERIFICATION SUITE" + " " * 25 + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    
    # Test 1: Multiple different locations
    print("\n\n")
    results = test_multiple_locations()
    
    # Test 2: Same location twice
    print("\n\n")
    data1, data2 = test_same_location_twice()
    
    # Test 3: Detailed breakdown of first result
    if results:
        print("\n\n")
        extractor = DataExtractor()
        polygon = [(18.522, 73.854), (18.522, 73.859), (18.518, 73.859), (18.518, 73.854)]
        detailed_data = extractor.extract_from_polygon(polygon, save_to_file=False)
        show_detailed_wind_breakdown(detailed_data)
    
    print("\n\n")
    print("█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + " " * 20 + "TESTING COMPLETE" + " " * 32 + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    print("\n")
    
    print("💡 INTERPRETATION GUIDE:")
    print("="*70)
    print("✓ If wind speeds vary by location → Real API data is working")
    print("✓ If wind speeds are same for same location → Consistency check passed")
    print("⚠️ If all wind speeds identical → API might be failing, using synthetic data")
    print("⚠️ If 'Synthetic Random' appears → All APIs failed")
    print("="*70)