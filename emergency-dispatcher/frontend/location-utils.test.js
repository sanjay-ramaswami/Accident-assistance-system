const test = require("node:test");
const assert = require("node:assert/strict");
require("./location-utils.js");

const { getMapCoordinates, getMapPresentation, getUnresolvedMapStatus, retainMapLocation } = globalThis.DispatcherLocationUtils;

test("approximate city geocodes show a city-scale pin and request a precise clue", () => {
  const presentation = getMapPresentation({
    location: "Kochi, Ernakulam, Kerala, India",
    confidence: "low",
    reason: "approximate_city_match",
  });

  assert.equal(presentation.zoom, 11);
  assert.match(presentation.markerLabel, /approximate city-level location/);
  assert.match(presentation.status, /low confidence/);
  assert.match(presentation.status, /neighbourhood, road, hospital, or landmark/);
});

test("high confidence geocodes retain the address-level map zoom", () => {
  const presentation = getMapPresentation({
    location: "128 Example Road",
    confidence: "high",
    reason: "google_match",
  });

  assert.equal(presentation.zoom, 15);
  assert.equal(presentation.markerLabel, "128 Example Road");
});

test("city-qualified Marine Drive pin stays approximate and asks for an incident clue", () => {
  const presentation = getMapPresentation({
    location: "Marine Drive, Kochi, Kerala, India",
    confidence: "low",
    reason: "approximate_place_match",
    source: "nominatim",
  });

  assert.equal(presentation.zoom, 14);
  assert.match(presentation.markerLabel, /incident point not confirmed/);
  assert.match(presentation.status, /low confidence/);
  assert.match(presentation.status, /nearby landmark/);
});

test("only validated provider coordinates can reach the map marker", () => {
  assert.deepEqual(getMapCoordinates("9.9784782", "76.2757727"), [9.9784782, 76.2757727]);
  assert.equal(getMapCoordinates("91", "76"), null);
  assert.equal(getMapCoordinates("NaN", "76"), null);
  assert.equal(getMapCoordinates(null, null), null);
});

test("ambiguous Marine Drive asks for a city or landmark", () => {
  const status = getUnresolvedMapStatus({
    location: "Marine Drive",
    mapReason: "ambiguous_location_without_city",
    locationNeedsConfirmation: true,
  });
  assert.match(status, /Which city/);
  assert.match(status, /nearby landmark/);
});

test("geocoder failure status names the safe category and next clue", () => {
  const status = getUnresolvedMapStatus({
    location: "Marine Drive, Kochi, Kerala, India",
    mapReason: "nominatim_error",
    mapSource: "text_fallback",
  });
  assert.match(status, /service unavailable \(nominatim_error\)/);
  assert.match(status, /nearby landmark/);
});

test("partial assessment update retains the last valid pin within its session", () => {
  const first = retainMapLocation(null, {
    lat: 9.978,
    lon: 76.276,
    location: "Marine Drive, Kochi",
    confidence: "low",
    revision: 3,
  }, "session-a");
  const next = retainMapLocation(first, { lat: null, lon: null, location: "Nearby clue pending", revision: 4 }, "session-a");

  assert.deepEqual(next.coordinates, [9.978, 76.276]);
  assert.equal(next.location, "Marine Drive, Kochi");
  assert.equal(next.retained, true);
  assert.equal(next.retentionReason, "partial_update");
});

test("new sessions clear retained pins and stale location revisions are ignored", () => {
  const first = retainMapLocation(null, {
    lat: 9.978,
    lon: 76.276,
    confidence: "high",
    revision: 8,
  }, "session-a");
  const stale = retainMapLocation(first, {
    lat: 12.0,
    lon: 77.0,
    confidence: "high",
    revision: 7,
  }, "session-a");
  const nextSession = retainMapLocation(first, { lat: null, lon: null }, "session-b");

  assert.deepEqual(stale.coordinates, [9.978, 76.276]);
  assert.equal(stale.retentionReason, "stale_update");
  assert.equal(nextSession, null);
});

test("lower-confidence geocoding cannot replace a confirmed pin but a confirmed correction can", () => {
  const confirmed = retainMapLocation(null, {
    lat: 9.978,
    lon: 76.276,
    confidence: "caller confirmed",
    revision: 4,
  }, "session-a");
  const approximate = retainMapLocation(confirmed, {
    lat: 10.0,
    lon: 76.3,
    confidence: "low",
    revision: 5,
  }, "session-a");
  const correction = retainMapLocation(confirmed, {
    lat: 9.98,
    lon: 76.28,
    confidence: "caller confirmed",
    revision: 6,
  }, "session-a");

  assert.deepEqual(approximate.coordinates, [9.978, 76.276]);
  assert.equal(approximate.retentionReason, "lower_confidence");
  assert.deepEqual(correction.coordinates, [9.98, 76.28]);
  assert.equal(correction.retained, false);
});

test("needs-confirmation is not ranked as caller-confirmed", () => {
  const { mapConfidenceRank } = globalThis.DispatcherLocationUtils;
  assert.equal(mapConfidenceRank("needs confirmation"), 1);
  assert.equal(mapConfidenceRank("confirmed by caller"), 4);
});
