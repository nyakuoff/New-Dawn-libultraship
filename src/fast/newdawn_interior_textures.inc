// Window and emissive textures for the New Dawn interior polish, by resource basename.
// { name, kind, a, b, c, d }  kind 1 = window: inset u, inset v, panes u, panes v.
//                              kind 2 = night glass, kind 3 = lamp/sign: tint rgb, light intensity (0 = no local light).
// No real window textures are listed: Majora's Mask interiors have almost none (see docs/INTERIOR_POLISH.md); the
// render-only windows come from the per-scene table in NewDawnPostFx.cpp. Kind 1 entries are supported for later use.
// Clock Tower faces (outside, seen across South Clock Town): lit after dusk. Tint is a warm white, no local light.
{ "gClockTowerClockFaceTopTex", 2, 1.0f, 0.85f, 0.6f, 0 },
{ "gClockTowerClockFaceBottomTex", 2, 1.0f, 0.85f, 0.6f, 0 },
{ "gClockTowerClockFacePanelBarTex", 2, 1.0f, 0.85f, 0.6f, 0 },
{ "gClockTowerMinuteRingFace1Tex", 2, 1.0f, 0.85f, 0.6f, 0 },
{ "gClockTowerMinuteRingFace2Tex", 2, 1.0f, 0.85f, 0.6f, 0 },
{ "gClockTowerMinuteRingFace3Tex", 2, 1.0f, 0.85f, 0.6f, 0 },
{ "gClockTowerSunAndMoonPanelSunTex", 2, 1.0f, 0.8f, 0.4f, 0 },
{ "gClockTowerSunAndMoonPanelMoonTex", 2, 0.7f, 0.8f, 1.0f, 0 },
// Lottery shop sign: always lit, with a small warm light.
{ "gLotteryShopSignBorderTex", 3, 1.0f, 0.75f, 0.4f, 0.6f },
{ "gLotteryShopSignTextTex", 3, 1.0f, 0.75f, 0.4f, 0 },
