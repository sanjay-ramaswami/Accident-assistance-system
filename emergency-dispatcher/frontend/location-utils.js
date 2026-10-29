(function (root) {
  function getMapCoordinates(lat, lon) {
    if (lat === null || lat === undefined || lon === null || lon === undefined) return null;
    const latitude = Number(lat);
    const longitude = Number(lon);
    if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) return null;
    if (latitude < -90 || latitude > 90 || longitude < -180 || longitude > 180) return null;
    return [latitude, longitude];
  }

  function mapConfidenceRank(confidence) {
    const value = String(confidence || "").toLowerCase();
    if (value.includes("needs confirmation") || value.includes("unverified")) return 1;
    if (value.includes("confirm")) return 4;
    if (value === "high" || value.includes("exact")) return 3;
    if (value === "medium" || value.includes("approximate")) return 2;
    if (value === "low" || value.includes("city")) return 1;
    return 0;
  }

  function retainMapLocation(previous, candidate, sessionId) {
    const prior = previous?.sessionId === sessionId ? previous : null;
    const candidateCoordinates = getMapCoordinates(candidate?.lat, candidate?.lon);
    const revision = Number(candidate?.revision || 0);
    if (prior && revision && prior.revision && revision < prior.revision) {
      return { ...prior, retained: true, retentionReason: "stale_update" };
    }
    if (candidateCoordinates) {
      const candidateRank = mapConfidenceRank(candidate?.confidence);
      if (prior && candidateRank < prior.quality) {
        return { ...prior, retained: true, retentionReason: "lower_confidence" };
      }
      return {
        ...candidate,
        sessionId,
        coordinates: candidateCoordinates,
        revision,
        quality: candidateRank,
        retained: false,
        retentionReason: null,
      };
    }
    return prior ? { ...prior, retained: true, retentionReason: "partial_update" } : null;
  }

  function getMapPresentation({ location, confidence, reason, source }) {
    const approximateCity = confidence === "low" && reason === "approximate_city_match";
    if (approximateCity) {
      return {
        zoom: 11,
        markerLabel: `${location} (approximate city-level location; not an exact address)`,
        status: `${location} · approximate city-level pin · low confidence. Ask for a neighbourhood, road, hospital, or landmark.`,
      };
    }
    if (reason === "approximate_place_match") {
      return {
        zoom: 14,
        markerLabel: `${location} (approximate city-qualified place; incident point not confirmed)`,
        status: `${location} · approximate city-qualified pin · low confidence. Ask for a nearby landmark, entrance, or exact incident point.`,
      };
    }
    return {
      zoom: confidence === "low" ? 11 : 15,
      markerLabel: location,
      status: `${location} · ${source || "map"} · ${confidence || "pending"} confidence`,
    };
  }

  function getUnresolvedMapStatus({ location, mapReason, mapSource, mapCandidates, locationNeedsConfirmation }) {
    if (mapReason === "ambiguous_location_without_city") {
      return "Marine Drive is ambiguous. Which city is it in, or what nearby landmark can you see?";
    }
    if (mapReason === "nominatim_error" || mapReason === "google_no_match_nominatim_error") {
      return `Location is unresolved: geocoding service unavailable (${mapReason}). Could you give a nearby landmark or another spelling?`;
    }
    if (mapReason === "nominatim_no_match" || mapReason === "google_no_match_nominatim_no_match") {
      return `Location is unresolved: Nominatim found no verified match (${mapReason}). Could you give a nearby landmark or street address?`;
    }
    if (locationNeedsConfirmation) return `Location candidate needs caller confirmation: ${location}.`;
    if (mapCandidates && mapCandidates.length) {
      return `No exact pin yet. Best text match: ${mapCandidates[0].display_name || location}.`;
    }
    if (mapSource === "text_fallback") {
      return `Location is unresolved: no verified coordinates (${mapReason || "no geocoder match"}). The map remains at the India overview. Could you give a nearby landmark or street address?`;
    }
    if (location && location !== "Pending") return `No exact pin yet. Current location text: ${location}.`;
    return "Waiting for a geocoded location.";
  }

  root.DispatcherLocationUtils = {
    getMapCoordinates,
    getMapPresentation,
    getUnresolvedMapStatus,
    mapConfidenceRank,
    retainMapLocation,
  };
})(typeof window !== "undefined" ? window : globalThis);
