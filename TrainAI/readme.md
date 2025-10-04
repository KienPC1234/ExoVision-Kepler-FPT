koi_kepmag: Độ sáng Kepler-band (magnitude) - đơn vị: mag (magnitude).
pl_radj: Bán kính hành tinh (Jupiter radii) - đơn vị: R_J (bán kính Jupiter, sau khi convert từ Earth radii nếu cần).
koi_impact: Tham số va chạm (impact parameter) - đơn vị: vô đơn vị (dimensionless).
pl_trandur: Thời gian quá cảnh (transit duration) - đơn vị: giờ (hours).
depth: Độ sâu quá cảnh (transit depth) - đơn vị: vô đơn vị (fraction, sau khi normalize từ ppm hoặc percent).
pl_orbper: Chu kỳ quỹ đạo (orbital period) - đơn vị: ngày (days).
st_teff: Nhiệt độ hiệu dụng của sao (stellar effective temperature) - đơn vị: K (Kelvin).
st_logg: Lực hấp dẫn bề mặt sao (stellar surface gravity, log scale) - đơn vị: dex (log10(cm/s²)).
st_rad: Bán kính sao (stellar radius) - đơn vị: R_Sun (bán kính Mặt Trời).
pl_insol: Thông lượng bức xạ (insolation flux) - đơn vị: F_Earth (tỷ lệ so với thông lượng Trái Đất, vô đơn vị tương đối).
pl_eqt: Nhiệt độ cân bằng hành tinh (equilibrium temperature) - đơn vị: K (Kelvin).
st_dist: Khoảng cách đến sao (stellar distance) - đơn vị: pc (parsec).
density_proxy: Proxy mật độ (derived: 1 / pl_radj³) - đơn vị: vô đơn vị (proxy, dựa trên bán kính Jupiter).
habitability_proxy: Proxy khả năng ở được (derived: pl_orbper * 0.7 / st_teff) - đơn vị: ngày/K (days per Kelvin, nhưng dùng như proxy vô đơn vị).
transit_shape_proxy: Proxy hình dạng quá cảnh (derived: depth / pl_trandur) - đơn vị: fraction/giờ (vô đơn vị tương đối).