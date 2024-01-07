-- Bicycle profile
api_version = 4
properties.force_split_edges = true

Set = require('lib/set')
Sequence = require('lib/sequence')
Handlers = require("lib/way_handlers")
find_access_tag = require("lib/access").find_access_tag
limit = require("lib/maxspeed").limit
Tags = require('lib/tags')
Measure = require("lib/measure")

function setup()
  -- To change the cycling speed, change both values below
  local default_speed = 14 -- In km/h 
  local walking_speed = 14 -- In km/h 
  allowed_sequential_ids = {}
  for id in io.lines("csv_filter_path") do -- this "csv_filter_path" will be modified on-the-fly by the python code. Do not change this.
    allowed_sequential_ids[tonumber(id)] = true
  end

  return {
    allowed_sequential_ids = allowed_sequential_ids,

    properties = {
      u_turn_penalty                = 20,
      traffic_light_penalty         = 0,
      weight_name                   = 'duration', -- routing based on shortest time
      --weight_name                   = 'routability',
      process_call_tagless_node     = false,
      max_speed_for_map_matching    = 110/3.6, -- kmph -> m/s
      use_turn_restrictions         = false,
      continue_straight_at_waypoint = false,
      mode_change_penalty           = 5,
    },

    default_mode              = mode.cycling,
    default_speed             = default_speed,
    walking_speed             = walking_speed,
    oneway_handling           = false,
    turn_penalty              = 0,
    turn_bias                 = 1,
    use_public_transport      = false,

    allowed_start_modes = Set {
      mode.cycling,
      mode.pushing_bike
    },

    barrier_blacklist = Set {
      'yes',
      'wall',
      'fence'
    },

    access_tag_whitelist = Set {
      'yes',
      'routing:bicycle',
      'bicycle',
      'permissive',
      'designated'
    },

    access_tag_blacklist = Set {
      'no',
      'private',
      'customers',
      'delivery',
      'agricultural',
      'forestry',
      'destination',
    },

    service_access_tag_blacklist = Set {
      --'private' -- default value in osrm default profile
    },

    restricted_access_tag_list = Set {
      'private',
      'delivery',
      'destination',
      'customers',
      'agricultural',
      'forestry'
    },

    restricted_highway_whitelist = Set {
      'trunk',
      'trunk_link',
      'primary',
      'primary_link',
      'secondary',
      'secondary_link',
      'tertiary',
      'tertiary_link',
      'residential',
      'living_street',
      'unclassified',
      'service',
      'footway',
      'bridleway',
      'track',
      'path',
      'cycleway',
      'pedestrian',
      'steps',
      'pier',
      'corridor',
    },

    -- tags disallow access to in combination with highway=service
    service_access_tag_blacklist = Set { },

    service_tag_forbidden = Set {
      'emergency_access'
    },

    service_penalties = {},

    construction_whitelist = Set {
      'no',
      'widening',
      'minor',
    },

    access_tags_hierarchy = Sequence {
      'routing:bicycle',
      'bicycle',
      'vehicle',
      'access'
    },

    restrictions = Set {
      'bicycle'
    },

    cycleways_normal = {
      ['lane'] = true,
      ['shared_lane'] = true,
      ['share_busway'] = true,
      ['track'] = true,
      ['shared'] = true,
      ['sharrow'] = true
    },

    cycleways_opposite = {
      ['opposite'] = true,
      ['opposite_lane'] = true,
      ['opposite_share_busway'] = true,
      ['opposite_track'] = true
    },

    service_penalties = {
      alley             = 0.8,
    },

    -- Speeds for the different types of cycling infrastructure: shared lane (sharrow), shared busway, lane and track
    -- Cycle values are from : Clarry, A., Faghih Imani, A., & Miller, E. J. (2019). Where we ride faster? Examining cycling speed using smartphone GPS data. Sustainable Cities and Society, 49, 101594. https://doi.org/10.1016/j.scs.2019.101594
    -- Note: The article does not consider shared bus lanes, so we set the same speed for those as for the shared lanes.
    bicycle_speeds = {
      shared_lane           = default_speed,-- + 0.130*3.6,
      shared                = default_speed,-- + 0.130*3.6,
      sharrow               = default_speed,-- + 0.130*3.6,
      share_busway          = default_speed,-- + 0.130*3.6,
      opposite              = default_speed,-- + 0.130*3.6,
      opposite_share_busway = default_speed,-- + 0.130*3.6,
      lane                  = default_speed,-- + 0.284*3.6,
      opposite_lane         = default_speed,-- + 0.284*3.6,
      track                 = default_speed,-- + 0.212*3.6,
      opposite_track        = default_speed,-- + 0.212*3.6,
      cycleway              = default_speed,-- + 0.212*3.6,
      -- The rest of these are left to the default speed.
      trunk = default_speed,
      trunk_link = default_speed,
      primary = default_speed,
      primary_link = default_speed,
      secondary = default_speed,
      secondary_link = default_speed,
      tertiary = default_speed,
      tertiary_link = default_speed,
      residential = default_speed,
      unclassified = default_speed,
      living_street = default_speed,
      road = default_speed,
      bridleway = default_speed,
      service = default_speed,
      path = default_speed
    },

    pedestrian_speeds = {
      footway = walking_speed,
      corridor = walking_speed,
      pedestrian = walking_speed,
      steps = 2
    },

    railway_speeds = {
      train = 10,
      railway = 10,
      subway = 10,
      light_rail = 10,
      monorail = 10,
      tram = 10
    },

    platform_speeds = {
      platform = walking_speed
    },

    amenity_speeds = {
      parking = 10,
      parking_entrance = 10
    },

    man_made_speeds = {
      pier = walking_speed
    },

    route_speeds = {
      ferry = 5
    },

    bridge_speeds = {
      movable = 5
    },

    surface_speeds = {
      asphalt = default_speed,
      ["cobblestone:flattened"] = 10,
      paving_stones = 12,
      compacted = 15,
      cobblestone = 6,
      unpaved = 15,
      fine_gravel = 15,
      gravel = 15,
      pebblestone = 6,
      ground = 6,
      dirt = 8,
      earth = 6,
      grass = 10,
      mud = 5,
      sand = 5,
      sett = 10
    },

    classes = Sequence {
        'ferry', 'tunnel'
    },

    -- Which classes should be excludable
    -- This increases memory usage so its disabled by default.
    excludable = Sequence {
--        Set {'ferry'}
    },

    tracktype_speeds = {
    },

    smoothness_speeds = {
    },

    avoid = Set {
      'impassable',
      --'construction'
    },
  }
end

function process_node(profile, node, result)
  -- parse access and barrier tags
  local highway = node:get_value_by_key("highway")
  local is_crossing = highway and highway == "crossing"

  local access = find_access_tag(node, profile.access_tags_hierarchy)
  if access and access ~= "" then
    -- access restrictions on crossing nodes are not relevant for
    -- the traffic on the road
    if profile.access_tag_blacklist[access] and not profile.restricted_access_tag_list[access] and not is_crossing then
      result.barrier = true
    end
  else
    local barrier = node:get_value_by_key("barrier")
    if barrier and "" ~= barrier then
      if profile.barrier_blacklist[barrier] then
        result.barrier = true
      end
    end
  end

  -- check if node is a traffic light
  local tag = node:get_value_by_key("highway")
  if tag and "traffic_signals" == tag then
    result.traffic_lights = true
  end
end

function handle_bicycle_tags(profile,way,result,data)
    -- initial routability check, filters out buildings, boundaries, etc
  data.route = way:get_value_by_key("route")
  data.man_made = way:get_value_by_key("man_made")
  data.railway = way:get_value_by_key("railway")
  data.amenity = way:get_value_by_key("amenity")
  data.public_transport = way:get_value_by_key("public_transport")
  data.bridge = way:get_value_by_key("bridge")

  if (not data.highway or data.highway == '') and
  (not data.route or data.route == '') and
  (not profile.use_public_transport or not data.railway or data.railway=='') and
  (not data.amenity or data.amenity=='') and
  (not data.man_made or data.man_made=='') and
  (not data.public_transport or data.public_transport=='') and
  (not data.bridge or data.bridge=='')
  then
    return false
  end
  --print(way:get_value_by_key("sequential_id"))
  if profile.allowed_sequential_ids[tonumber(way:get_value_by_key("sequential_id"))] == nil then
    return false
  end

  -- access

  data.forward_access, data.backward_access =
    Tags.get_forward_backward_by_set(way,data,profile.access_tags_hierarchy)

  data.access = find_access_tag(way, profile.access_tags_hierarchy)
  if data.access and profile.access_tag_blacklist[data.access] and not profile.restricted_highway_whitelist[data.highway] then
    return false
  end
  if profile.restricted_highway_whitelist[data.highway] then
    if profile.restricted_access_tag_list[data.forward_access] then
        result.forward_restricted = true
    end

    if profile.restricted_access_tag_list[data.backward_access] then
        result.backward_restricted = true
    end

  end

  -- other tags
  data.junction = way:get_value_by_key("junction")
  data.maxspeed = Measure.get_max_speed(way:get_value_by_key ("maxspeed")) or 0
  data.maxspeed_forward = Measure.get_max_speed(way:get_value_by_key("maxspeed:forward")) or 0
  data.maxspeed_backward = Measure.get_max_speed(way:get_value_by_key("maxspeed:backward")) or 0
  data.barrier = way:get_value_by_key("barrier")
  data.oneway = way:get_value_by_key("oneway")
  data.oneway_bicycle = way:get_value_by_key("oneway:bicycle")
  data.cycleway = way:get_value_by_key("cycleway")
  data.cycleway_left = way:get_value_by_key("cycleway:left")
  data.cycleway_right = way:get_value_by_key("cycleway:right")
  data.cycleway_both = way:get_value_by_key("cycleway:both")
  data.duration = way:get_value_by_key("duration")
  data.service = way:get_value_by_key("service")
  data.foot = way:get_value_by_key("foot")
  data.foot_forward = way:get_value_by_key("foot:forward")
  data.foot_backward = way:get_value_by_key("foot:backward")
  data.bicycle = way:get_value_by_key("bicycle")
  data.bicycle_routing = way:get_value_by_key("routing:bicycle")

  speed_handler(profile,way,result,data)

  --oneway_handler(profile,way,result,data)

  cycleway_handler(profile,way,result,data)

  bike_push_handler(profile,way,result,data)

  -- After checking speed, one way, cycleway facilities and bike pushing, we are now ready to
  -- set the rate which will impact the 'weight' and thus routing.
  --rate_handler(profile,way,result,data)

  -- Directly set the rate equal to the speed for all links.
  result.forward_rate = result.forward_speed
  result.backward_rate = result.backward_speed

  -- maxspeed
  limit( result, data.maxspeed, data.maxspeed_forward, data.maxspeed_backward )

  -- not routable if no speed assigned
  -- this avoid assertions in debug builds
  if result.forward_speed <= 0 and result.duration <= 0 then
    result.forward_mode = mode.inaccessible
  end
  if result.backward_speed <= 0 and result.duration <= 0 then
    result.backward_mode = mode.inaccessible
  end

  return true
end



function speed_handler(profile,way,result,data)

  data.way_type_allows_pushing = false

  -- speed
  local bridge_speed = profile.bridge_speeds[data.bridge]
  if (bridge_speed and bridge_speed > 0) then
    data.highway = data.bridge
    if data.duration and durationIsValid(data.duration) then
      result.duration = math.max( parseDuration(data.duration), 1 )
    end
    result.forward_speed = bridge_speed
    result.backward_speed = bridge_speed
    data.way_type_allows_pushing = true
  elseif profile.route_speeds[data.route] then
    -- ferries (doesn't cover routes tagged using relations)
    result.forward_mode = mode.ferry
    result.backward_mode = mode.ferry
    if data.duration and durationIsValid(data.duration) then
      result.duration = math.max( 1, parseDuration(data.duration) )
    else
       result.forward_speed = profile.route_speeds[data.route]
       result.backward_speed = profile.route_speeds[data.route]
    end
  -- railway platforms (old tagging scheme)
  elseif data.railway and profile.platform_speeds[data.railway] then
    result.forward_speed = profile.platform_speeds[data.railway]
    result.backward_speed = profile.platform_speeds[data.railway]
    data.way_type_allows_pushing = true
  -- public_transport platforms (new tagging platform)
  elseif data.public_transport and profile.platform_speeds[data.public_transport] then
    result.forward_speed = profile.platform_speeds[data.public_transport]
    result.backward_speed = profile.platform_speeds[data.public_transport]
    data.way_type_allows_pushing = true
  -- railways
  elseif profile.use_public_transport and data.railway and profile.railway_speeds[data.railway] and profile.access_tag_whitelist[data.access]
then
    result.forward_mode = mode.train
    result.backward_mode = mode.train
    result.forward_speed = profile.railway_speeds[data.railway]
    result.backward_speed = profile.railway_speeds[data.railway]
  elseif data.amenity and profile.amenity_speeds[data.amenity] then
    -- parking areas
    result.forward_speed = profile.amenity_speeds[data.amenity]
    result.backward_speed = profile.amenity_speeds[data.amenity]
    data.way_type_allows_pushing = true
  elseif profile.bicycle_speeds[data.highway] then
    -- regular ways
    result.forward_speed = profile.bicycle_speeds[data.highway]
    result.backward_speed = profile.bicycle_speeds[data.highway]
    data.way_type_allows_pushing = true
  elseif data.access and profile.access_tag_whitelist[data.access]  then
    -- unknown way, but valid access tag
    result.forward_speed = profile.default_speed
    result.backward_speed = profile.default_speed
    data.way_type_allows_pushing = true
  end
end

function oneway_handler(profile,way,result,data)
  -- oneway
  data.implied_oneway = data.junction == "roundabout" or data.junction == "circular" or data.highway == "motorway"
  data.reverse = false
  data.bicycle_oneway = false

  if data.oneway_bicycle == "yes" or data.oneway_bicycle == "1" or data.oneway_bicycle == "true" then
    result.backward_mode = mode.inaccessible
    data.bicycle_oneway = true
  elseif data.oneway_bicycle == "no" or data.oneway_bicycle == "0" or data.oneway_bicycle == "false" then
    data.bicycle_oneway = false
  elseif data.oneway_bicycle == "-1" then
    result.forward_mode = mode.inaccessible
    data.reverse = true
  elseif data.oneway == "yes" or data.oneway == "1" or data.oneway == "true" then
    result.backward_mode = mode.inaccessible
    data.bicycle_oneway = true
  elseif data.oneway == "no" or data.oneway == "0" or data.oneway == "false" then
    data.bicycle_oneway = false
  elseif data.oneway == "-1" then
    result.forward_mode = mode.inaccessible
    data.reverse = true
  elseif data.implied_oneway then
    result.backward_mode = mode.inaccessible
    data.bicycle_oneway = true
  end
end

function cycleway_handler(profile, way, result, data)
  local is_twoway = result.forward_mode ~= mode.inaccessible and result.backward_mode ~= mode.inaccessible and not data.implied_oneway

  data.cycleway_forward = nil
  data.cycleway_backward = nil

  if is_twoway then
    -- There is a cycling facility, probably in both directions (if not, it should have been tagged with :left and :right tags)
    if (data.cycleway and profile.cycleways_normal[data.cycleway]) or
       (data.cycleway_both and profile.cycleways_normal[data.cycleway_both]) then
      data.cycleway_forward = data.cycleway or data.cycleway_both
      data.cycleway_backward = data.cycleway or data.cycleway_both
    end

    -- A *normal* cycling facility on the right means that there is a cycling facility in the forward direction.
    if data.cycleway_right and profile.cycleways_normal[data.cycleway_right] then
      data.cycleway_forward = data.cycleway_right
    -- An *opposite* cycling facility on the right means there is a cycleway in the backward direction.
    elseif data.cycleway_right and profile.cycleways_opposite[data.cycleway_right] then
      data.cycleway_backward = data.cycleway_right
    end

    -- An *opposite* cycling facility on the left means there is a cycling facility in the forward direction.
    if data.cycleway_left and profile.cycleways_opposite[data.cycleway_left] then
      -- We need to check if it's a one way before knowing if the cycleway is in the forward or backward direction.
      -- For example, imagine a road with an opposite_lane on the left.
      -- If the road is a two-way car road, the "opposite" in "opposite_lane" means that this lane is in the opposite direction
      -- of the nearby traffic. Therefore it goes in the forward direction.
      -- However, if this road is a one-way car road, then this cycling lane is contra-flow and goes *backward*.
      -- I am not sure if this is the right way to handle this, but it seems to work.
      -- In any case, cycleway:left=opposite_lane should probably be replaced with cycleway:left=lane + cycleway:left:oneway=* instead.
      -- See https://wiki.openstreetmap.org/wiki/Talk:Key:cycleway:left#Why_opposite_lane_is_supposed_to_be_invalid_value? for more explanations
      if data.bicycle_oneway then
        data.cycleway_forward = data.cycleway_left
      else
        data.cycleway_backward = data.cycleway_left
      end
    -- A *normal* cycling facility on the left means that there is a cycling facility in the backward direction.
    elseif data.cycleway_left and profile.cycleways_normal[data.cycleway_left] then
      data.cycleway_backward = data.cycleway_left
    end

  else -- one way. Refer to https://github.com/Project-OSRM/osrm-backend/issues/4943#issuecomment-371832446 for explanation
    local has_twoway_cycleway = (data.cycleway and profile.cycleways_opposite[data.cycleway]) or (data.cycleway_both and profile.cycleways_opposite[data.cycleway_both]) or (data.cycleway_right and profile.cycleways_opposite[data.cycleway_right]) or (data.cycleway_left and profile.cycleways_opposite[data.cycleway_left])
    local has_opposite_cycleway = (data.cycleway_left and profile.cycleways_opposite[data.cycleway_left]) or (data.cycleway_right and profile.cycleways_opposite[data.cycleway_right])
    local has_oneway_cycleway = (data.cycleway and profile.cycleways_normal[data.cycleway]) or (data.cycleway_both and profile.cycleways_normal[data.cycleway_both]) or (data.cycleway_right and profile.cycleways_normal[data.cycleway_right]) or (data.cycleway_left and profile.cycleways_normal[data.cycleway_left])

    -- TODO: Clean up this mess

    -- Two-way cycleway checks
    if has_twoway_cycleway then
      if data.cycleway and profile.cycleways_opposite[data.cycleway] then
        data.cycleway_backward = data.cycleway; data.cycleway_forward = data.cycleway
      elseif data.cycleway_both and profile.cycleways_opposite[data.cycleway_both] then
        data.cycleway_backward = data.cycleway_both; data.cycleway_forward = data.cycleway_both
      elseif data.cycleway_right and profile.cycleways_opposite[data.cycleway_right] then
        data.cycleway_backward = data.cycleway_right; data.cycleway_forward = data.cycleway_right
      elseif data.cycleway_left and profile.cycleways_opposite[data.cycleway_left] then
        data.cycleway_backward = data.cycleway_left; data.cycleway_forward = data.cycleway_left
      end
    -- Opposite cycleway checks
    elseif has_opposite_cycleway then
      cycling_facility = nil
      if data.cycleway_left and profile.cycleways_opposite[data.cycleway_left] then
        cycling_facility = cycleway_left
      elseif data.cycleway_right and profile.cycleways_opposite[data.cycleway_right] then
        cycling_facility = cycleway_right
      end

      if not data.reverse then
        data.cycleway_backward = cycling_facility
      else
        data.cycleway_forward = cycling_facility
      end
    -- Oneway cycleway checks
    elseif has_oneway_cycleway then
      cycling_facility = nil
      if data.cycleway and profile.cycleways_normal[data.cycleway] then
        cycling_facility = data.cycleway
      elseif data.cycleway_both and profile.cycleways_normal[data.cycleway_both] then
        cycling_facility = data.cycleway_both
      elseif data.cycleway_right and profile.cycleways_normal[data.cycleway_right] then
        cycling_facility = data.cycleway_right
      elseif data.cycleway_left and profile.cycleways_normal[data.cycleway_left] then
        cycling_facility = data.cycleway_left
      end

      if not data.reverse then
        data.cycleway_forward = cycling_facility
      else
        data.cycleway_backward = cycling_facility
      end
    end
  end

  if data.cycleway_backward ~= nil then
    result.backward_mode = mode.cycling
    result.backward_speed = profile.bicycle_speeds[data.cycleway_backward]
  end

  if data.cycleway_forward ~= nil then
    result.forward_mode = mode.cycling
    result.forward_speed = profile.bicycle_speeds[data.cycleway_forward]
  end
end

function bike_push_handler(profile,way,result,data)
  -- pushing bikes - if no other mode found
  if result.forward_mode == mode.inaccessible or result.backward_mode == mode.inaccessible or
    result.forward_speed == -1 or result.backward_speed == -1 then
    if data.foot ~= 'no' then
      local push_forward_speed = nil
      local push_backward_speed = nil

      if profile.pedestrian_speeds[data.highway] then
        push_forward_speed = profile.pedestrian_speeds[data.highway]
        push_backward_speed = profile.pedestrian_speeds[data.highway]
      elseif data.man_made and profile.man_made_speeds[data.man_made] then
        push_forward_speed = profile.man_made_speeds[data.man_made]
        push_backward_speed = profile.man_made_speeds[data.man_made]
      else
        if data.foot == 'yes' then
          push_forward_speed = profile.walking_speed
          if not data.implied_oneway then
            push_backward_speed = profile.walking_speed
          end
        elseif data.foot_forward == 'yes' then
          push_forward_speed = profile.walking_speed
        elseif data.foot_backward == 'yes' then
          push_backward_speed = profile.walking_speed
        elseif data.way_type_allows_pushing then
          push_forward_speed = profile.walking_speed
          if not data.implied_oneway then
            push_backward_speed = profile.walking_speed
          end
        end
      end

      if push_forward_speed and (result.forward_mode == mode.inaccessible or result.forward_speed == -1) then
        result.forward_mode = mode.pushing_bike
        result.forward_speed = push_forward_speed
      end
      if push_backward_speed and (result.backward_mode == mode.inaccessible or result.backward_speed == -1)then
        result.backward_mode = mode.pushing_bike
        result.backward_speed = push_backward_speed
      end

    end

  end

  -- dismount
  if data.bicycle == "dismount" then
    result.forward_mode = mode.pushing_bike
    result.backward_mode = mode.pushing_bike
    result.forward_speed = profile.walking_speed
    result.backward_speed = profile.walking_speed
  end

  --if data.bicycle_routing == "use_sidepath" then
  --  result.forward_mode = mode.inaccessible
  --  result.backward_mode = mode.inaccessible
  --end
end

function rate_handler(profile, way, result, data)
  if profile.properties.weight_name == 'cyclability' then
    -- The division by 3.6 is to convert km/h to m/s.
    -- Values are from the article by Morency, Grante and Bourdeau: Impacts of Cyclability Features on Optimal Cycling Route

    if data.highway == "primary" or
      data.highway == "primary_link" or
      data.highway == "secondary" or
      data.highway == "secondary_link" then

      result.forward_rate = result.forward_speed / 3.6 * (1/7.954)
      result.backward_rate = result.backward_speed / 3.6 * (1/7.954)
    elseif data.highway == "tertiary" or
      data.highway == "tertiary_link" or
      data.highway == "residential" or
      data.highway == "road" or
      data.highway == "unclassified" or
      data.highway == "service" then

      result.forward_rate = result.forward_speed / 3.6 * (1/2.3865)
      result.backward_rate = result.backward_speed / 3.6 * (1/2.3865)
    elseif data.highway == "living_street" then
      result.forward_rate = result.forward_speed / 3.6 * (1/1.294)
      result.backward_rate = result.backward_speed / 3.6 * (1/1.294)
    elseif data.highway == "pedestrian" or
      data.highway == "track" or
      data.highway == "path" or
      data.highway == "cycleway" -- dedicated cycle way
      then
      result.forward_rate = result.forward_speed / 3.6 * (1/1.)
      result.backward_rate = result.backward_speed / 3.6 * (1/1.)
    end

    -- Classification used in the article mentionned above.
    group_1_cycleways =  {['lane'] = true, ['shared_lane'] = true, ['share_busway'] = true, ['opposite'] = true, ['opposite_lane'] = true, ['opposite_share_busway'] = true, ['sharrow'] = true}
    group_2_cycleways = {['track'] = true, ['opposite_track'] = true}

    -- Group 1
    if group_1_cycleways[data.cycleway_forward] then
      result.forward_rate = result.forward_speed / 3.6 * (1/1.294)
    end
    if group_1_cycleways[data.cycleway_backward] then
      result.backward_rate = result.backward_speed / 3.6 * (1/1.294)
    end

    -- Group 2
    if group_2_cycleways[data.cycleway_forward] then

      result.forward_rate = result.forward_speed / 3.6
    end
    if group_2_cycleways[data.cycleway_backward] then
      result.backward_rate = result.backward_speed / 3.6
    end

    --print("Forward: ", data.cycleway_forward)
    --print("Backward: ", data.cycleway_backward)
  end
end

function process_way(profile, way, result)
  -- the initial filtering of ways based on presence of tags
  -- affects processing times significantly, because all ways
  -- have to be checked.
  -- to increase performance, prefetching and initial tag check
  -- is done directly instead of via a handler.

  -- in general we should try to abort as soon as
  -- possible if the way is not routable, to avoid doing
  -- unnecessary work. this implies we should check things that
  -- commonly forbids access early, and handle edge cases later.

  -- data table for storing intermediate values during processing

  local data = {
    -- prefetch tags
    highway = way:get_value_by_key('highway'),

    route = nil,
    man_made = nil,
    railway = nil,
    amenity = nil,
    public_transport = nil,
    bridge = nil,

    access = nil,

    junction = nil,
    maxspeed = nil,
    maxspeed_forward = nil,
    maxspeed_backward = nil,
    barrier = nil,
    oneway = nil,
    oneway_bicycle = nil,
    cycleway = nil,
    cycleway_left = nil,
    cycleway_right = nil,
    duration = nil,
    service = nil,
    foot = nil,
    foot_forward = nil,
    foot_backward = nil,
    bicycle = nil,

    way_type_allows_pushing = false,
    cycleway_forward = nil,
    cycleway_forward = nil,
    is_twoway = true,
    reverse = false,
    implied_oneway = false
  }

  local handlers = Sequence {
    -- set the default mode for this profile. if can be changed later
    -- in case it turns we're e.g. on a ferry
    WayHandlers.default_mode,

    -- check various tags that could indicate that the way is not
    -- routable. this includes things like status=impassable,
    -- toll=yes and oneway=reversible
    WayHandlers.blocked_ways,
    WayHandlers.avoid_ways,

    -- our main handler
    handle_bicycle_tags,

    WayHandlers.access,

    -- compute speed taking into account way type, maxspeed tags, etc.
    -- David: commented out because for now we won't take surface type into account.
    --WayHandlers.surface,

    -- handle turn lanes and road classification, used for guidance
    WayHandlers.classification,

    -- handle allowed start/end modes
    WayHandlers.startpoint,

    -- handle roundabouts
    WayHandlers.roundabouts,
    WayHandlers.penalties,

    -- set name, ref and pronunciation
    WayHandlers.names,

    -- set classes
    WayHandlers.classes,

    -- set weight properties of the way
    WayHandlers.weights
  }

  WayHandlers.run(profile, way, result, data, handlers)
end

function process_turn(profile, turn)

  turn.duration = profile.turn_penalty
  -- compute turn penalty as angle^2, with a left/right bias
  local normalized_angle = turn.angle / 90.0
  if normalized_angle >= 0.0 then
    turn.duration = normalized_angle * normalized_angle * profile.turn_penalty / profile.turn_bias
  else
    turn.duration = normalized_angle * normalized_angle * profile.turn_penalty * profile.turn_bias
  end

  if turn.is_u_turn then
    turn.duration = turn.duration + profile.properties.u_turn_penalty
  end

  if turn.has_traffic_light then
     turn.duration = turn.duration + profile.properties.traffic_light_penalty
  end
  if profile.properties.weight_name == 'duration' then
    turn.weight = turn.duration
  end
  if turn.source_mode == mode.cycling and turn.target_mode ~= mode.cycling then
    turn.weight = turn.weight + profile.properties.mode_change_penalty
  end
  if profile.properties.weight_name == 'duration' then--
    -- penalize turns from non-local access only segments onto local access only tags
    if not turn.source_restricted and turn.target_restricted then
        turn.weight = turn.weight + 3000
    end
  end
end

return {
  setup = setup,
  process_way = process_way,
  process_node = process_node,
  process_turn = process_turn,
}
