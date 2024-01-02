#include "osrm/match_parameters.hpp"
#include "osrm/nearest_parameters.hpp"
#include "osrm/route_parameters.hpp"
#include "osrm/table_parameters.hpp"
#include "osrm/trip_parameters.hpp"

#include "osrm/coordinate.hpp"
#include "osrm/engine_config.hpp"
#include "osrm/json_container.hpp"

#include "osrm/osrm.hpp"
#include "osrm/status.hpp"

#include "json_renderer.hpp"

#include <exception>
#include <iostream>
#include <string>
#include <utility>

#include <cstdlib>
#include <fstream>
#include <sstream>

struct Request {
    int id;
    float start_lat;
    float start_lon;
    float end_lat;
    float end_lon;
};

int main(int argc, const char *argv[])
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " data.osrm od_data.csv\n";
        return EXIT_FAILURE;
    }

    using namespace osrm;

    // Configure based on a .osrm base path
    EngineConfig config;

    config.storage_config = {argv[1]};
    config.use_shared_memory = false;
    config.algorithm = EngineConfig::Algorithm::MLD;

    // Routing machine with several services (such as Route, Table, Nearest, Trip, Match)
    static const OSRM osrm{config};

    std::ifstream csv_input_file(argv[2]);

    // Make sure the file is open
    if (!csv_input_file.is_open()) throw std::runtime_error("Could not open CSV file");

    util::json::Array json_output;

    // Read data, line by line
    std::string line;

    while(std::getline(csv_input_file, line))
    {
        struct Request input_request;
        std::stringstream line_stream(line);
        std::string temp;
        std::getline(line_stream, temp, ',');
        input_request.id = stoi(temp);
        std::getline(line_stream, temp, ',');
        input_request.start_lon = stof(temp);
        std::getline(line_stream, temp, ',');
        input_request.start_lat = stof(temp);
        std::getline(line_stream, temp, ',');
        input_request.end_lon = stof(temp);
        std::getline(line_stream, temp, ',');
        input_request.end_lat = stof(temp);
        std::getline(line_stream, temp, ',');

        RouteParameters params;

        params.coordinates.push_back({util::FloatLongitude{input_request.start_lon}, util::FloatLatitude{input_request.start_lat}});
        params.coordinates.push_back({util::FloatLongitude{input_request.end_lon}, util::FloatLatitude{input_request.end_lat}});
        params.annotations_type = RouteParameters::AnnotationsType::Distance | RouteParameters::AnnotationsType::Datasources | RouteParameters::AnnotationsType::Nodes;

        // Response is in JSON format
        engine::api::ResultT result = util::json::Object();

        // Execute routing request, this does the heavy lifting
        const auto status = osrm.Route(params, result);

        util::json::Object json_result = result.get<util::json::Object>();
        json_result.values.erase("waypoints"); // get rid of data we don't need

        json_result.values["id"] = input_request.id;
        json_output.values.push_back(json_result);
    }

    csv_input_file.close();

    // Convert the JSON objects to a string for printing to stdout.
    std::string json_string;
    util::json::Renderer renderer(json_string);
    renderer(json_output);
    std::cout << json_string;
}
