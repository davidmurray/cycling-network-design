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

#ifdef __APPLE__
#include <dispatch/dispatch.h>
#endif

struct Request {
    int id;
    float latorig;
    float lonorig;
    float latdest;
    float londest;
};

int main(int argc, const char *argv[])
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " data.osrm od_data.csv\n";
        return EXIT_FAILURE;
    }

    using namespace osrm;

    // Configure based on a .osrm base path, and no datasets in shared mem from osrm-datastore
    EngineConfig config;

    config.storage_config = {argv[1]};
    config.use_shared_memory = false;
    config.algorithm = EngineConfig::Algorithm::MLD;

    // Routing machine with several services (such as Route, Table, Nearest, Trip, Match)
    static const OSRM osrm{config};

    std::ifstream csv_input_file(argv[2]);

    // Make sure the file is open
    if (!csv_input_file.is_open()) throw std::runtime_error("Could not open CSV file");

    #ifdef __APPLE__
    dispatch_group_t grp = dispatch_group_create();
    dispatch_queue_t work_queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0);
    // This queue is used when writing to the json_output array. It serves basically like a lock.
    dispatch_queue_t lock_queue = dispatch_queue_create("com.davidmurray.lock_queue", NULL);
    #endif

    __block util::json::Array json_output;


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
        input_request.lonorig = stof(temp);
        std::getline(line_stream, temp, ',');
        input_request.latorig = stof(temp);
        std::getline(line_stream, temp, ',');
        input_request.londest = stof(temp);
        std::getline(line_stream, temp, ',');
        input_request.latdest = stof(temp);
        std::getline(line_stream, temp, ',');

        RouteParameters params;

        params.coordinates.push_back({util::FloatLongitude{input_request.lonorig}, util::FloatLatitude{input_request.latorig}});
        params.coordinates.push_back({util::FloatLongitude{input_request.londest}, util::FloatLatitude{input_request.latdest}});
        params.annotations_type = RouteParameters::AnnotationsType::Distance | RouteParameters::AnnotationsType::Datasources | RouteParameters::AnnotationsType::Nodes;
	    params.geometries = RouteParameters::GeometriesType::Polyline6;
	    params.overview = RouteParameters::OverviewType::Full;

        dispatch_group_async(grp, work_queue, ^{
            // Response is in JSON format
            engine::api::ResultT result = util::json::Object();

            // Execute routing request, this does the heavy lifting
            const auto status = osrm.Route(params, result);

            util::json::Object json_result = result.get<util::json::Object>();
            json_result.values.erase("waypoints"); // get rid of data we don't need

            json_result.values["id"] = input_request.id;
            dispatch_group_async(grp, lock_queue, ^{
                json_output.values.push_back(json_result);
            });

            // Render JSON response as string
            //std::string json_string;
            //osrm::util::json::render(json_string, json_result);

            // Implement JSON Text Sequences according to https://datatracker.ietf.org/doc/html/rfc7464
            //const char record_separator = (char)(30); // 30 is the ASCII decimal value for "record separator"
            //const char line_feed = (char)(10); // 30 is the ASCII decimal value for "record separator"
            //std::cout << record_separator << json_string << line_feed;

            //std::cout << json_string << ",";
        });
    }

    csv_input_file.close();

    // Wait until all asynchronous path requests have completed
    dispatch_group_wait(grp, DISPATCH_TIME_FOREVER);

    // Convert the JSON objects to a string for printing to stdout.
    std::string json_string;
    util::json::Renderer renderer(json_string);
    renderer(json_output);
    std::cout << json_string;
}
