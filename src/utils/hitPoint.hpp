#pragma once

#include <stdint.h>
#include <vector>
#include "../common/common.h"
#include "../common/geometry.hpp"

struct HitPoint {
    utils::Vector3 p, weight, flux_i, flux_d, d, norm;  // hitPoint, diffuse weight collected during ray tracing pass, indirect flux, direct flux, ray dir, normal vector
    uint32_t n = 0, delta_n = 0; // N and M in the paper
    const BRDF *brdf;
    double r_sqr = 0;
    bool valid = false;
};

struct HitPointKDNode {
    uint32_t idx;
    uint32_t left, right;
    utils::Vector3 p0, p1; // bbox
    double max_r_sqr;
};

class HitPointKDTree {
    uint32_t root;
    std::vector<HitPointKDNode> nodes;
    std::vector<uint32_t> hit_pnt_idx;
    uint32_t m_depth;

    uint32_t buildTree(uint32_t l, uint32_t r, SplitAxis axis, uint32_t depth = 1) {
        uint32_t idx = nodes.size();
        nodes.emplace_back(HitPointKDNode());
        nodes[idx].p0 = utils::Vector3(INF, INF, INF), nodes[idx].p1 = utils::Vector3(-INF, -INF, -INF);
        nodes[idx].max_r_sqr = 0;
        for (uint32_t i = l; i <= r; ++i) {
            nodes[idx].p0 = min(nodes[idx].p0, hit_pnts[hit_pnt_idx[i]].p);
            nodes[idx].p1 = max(nodes[idx].p1, hit_pnts[hit_pnt_idx[i]].p);
            nodes[idx].max_r_sqr = std::max(nodes[idx].max_r_sqr, hit_pnts[hit_pnt_idx[i]].r_sqr);
        }
        uint32_t mid = (l + r) >> 1;
        if (axis == X_AXIS) {
            std::nth_element(hit_pnt_idx.begin() + l, hit_pnt_idx.begin() + mid, std::next(hit_pnt_idx.begin() + r),
                             [this](const uint32_t &a, const uint32_t &b) {
                                 return hit_pnts[a].p.x() < hit_pnts[b].p.x();
                             });
        } else if (axis == Y_AXIS) {
            std::nth_element(hit_pnt_idx.begin() + l, hit_pnt_idx.begin() + mid, std::next(hit_pnt_idx.begin() + r),
                             [this](const uint32_t &a, const uint32_t &b) {
                                 return hit_pnts[a].p.y() < hit_pnts[b].p.y();
                             });
        } else {
            std::nth_element(hit_pnt_idx.begin() + l, hit_pnt_idx.begin() + mid, std::next(hit_pnt_idx.begin() + r),
                             [this](const uint32_t &a, const uint32_t &b) {
                                 return hit_pnts[a].p.z() < hit_pnts[b].p.z();
                             });
        }
        nodes[idx].idx = hit_pnt_idx[mid];
        if (l < mid)
            nodes[idx].left = buildTree(l, mid - 1, static_cast<SplitAxis>((static_cast<uint32_t>(axis) + 1) % 3),
                                        depth + 1);
        else
            nodes[idx].left = NULL_NODE;
        if (mid < r)
            nodes[idx].right = buildTree(mid + 1, r, static_cast<SplitAxis>((static_cast<uint32_t>(axis) + 1) % 3),
                                         depth + 1);
        else
            nodes[idx].right = NULL_NODE;
        if (nodes[idx].left == NULL_NODE && nodes[idx].right == NULL_NODE && depth > m_depth)
            m_depth = depth;
        return idx;
    }

public:
    std::vector<HitPoint> hit_pnts;

    HitPointKDTree(const uint32_t size) : hit_pnts(size), m_depth(0) {
        hit_pnt_idx.reserve(size);
        nodes.reserve(size);
        for (size_t i = 0; i < size; ++i)
            hit_pnt_idx.emplace_back(i);
        //root = buildTree(0, hit_pnts.size() - 1, X_AXIS);
    }

    void updateHitPointStats() {
        for (auto &x : hit_pnts) {
            if (x.n == 0) {
                x.n += x.delta_n;
                x.delta_n = 0;
                continue;
            }
            double factor = (x.n + SPPM_ALPHA * x.delta_n) / (x.n + x.delta_n);
            x.n += x.delta_n;
            x.delta_n = 0;
            x.r_sqr *= factor;
            x.flux_i *= factor;
        }
    }

    void setInitialRadiusHeuristic(int w, int h) {
        utils::Vector3 pp0(INF, INF, INF);
        utils::Vector3 pp1 = -pp0;
        for (auto &x : hit_pnts) {
            pp0 = utils::min(pp0, x.p);
            pp1 = utils::max(pp1, x.p);
        }
        auto box = pp1 - pp0;
        double irad = (box.x() + box.y() + box.z()) / (3. * (w + h));
        for (auto &x : hit_pnts) {
            x.r_sqr = irad * irad;
        }
    }

    void clearPreviousWeight() {
        for (auto &x : hit_pnts)
            x.weight = utils::Vector3();
    }

    void initializeTree() {
        auto sz = nodes.size();
        nodes.clear();
        nodes.reserve(sz);
        root = buildTree(0, hit_pnts.size() - 1, X_AXIS);
    }

    std::vector<uint32_t> testUpdate(uint32_t node, const utils::Vector3 &photon) const {
        if (node == NULL_NODE)
            return std::vector<uint32_t>();
        double dmin = 0;
        if (photon.x() > nodes[node].p1.x())
            dmin += (photon.x() - nodes[node].p1.x()) * (photon.x() - nodes[node].p1.x());
        if (photon.x() < nodes[node].p0.x())
            dmin += (photon.x() - nodes[node].p0.x()) * (photon.x() - nodes[node].p0.x());
        if (photon.y() > nodes[node].p1.y())
            dmin += (photon.y() - nodes[node].p1.y()) * (photon.y() - nodes[node].p1.y());
        if (photon.y() < nodes[node].p0.y())
            dmin += (photon.y() - nodes[node].p0.y()) * (photon.y() - nodes[node].p0.y());
        if (photon.z() > nodes[node].p1.z())
            dmin += (photon.z() - nodes[node].p1.z()) * (photon.z() - nodes[node].p1.z());
        if (photon.z() < nodes[node].p0.z())
            dmin += (photon.z() - nodes[node].p0.z()) * (photon.z() - nodes[node].p0.z());
        if (dmin > nodes[node].max_r_sqr)
            return std::vector<uint32_t>();
        auto &pnt = hit_pnts[nodes[node].idx];
        std::vector<uint32_t> res;
        if ((photon - pnt.p).len2() <= pnt.r_sqr && pnt.valid) {
            res.emplace_back(nodes[node].idx);
        }
        decltype(res) left, right;
        // recursively update children
        if (nodes[node].left != NULL_NODE)
            left = testUpdate(nodes[node].left, photon);
        if (nodes[node].right != NULL_NODE)
            right = testUpdate(nodes[node].right, photon);
        res.insert(res.end(), left.begin(), left.end());
        res.insert(res.end(), right.begin(), right.end());
        return res;
    }

    void update(uint32_t node, const utils::Vector3 &photon, const utils::Vector3 &weight, const utils::Vector3 &d) {
        if (node == NULL_NODE)
            return;
        double dmin = 0;
        if (photon.x() > nodes[node].p1.x())
            dmin += (photon.x() - nodes[node].p1.x()) * (photon.x() - nodes[node].p1.x());
        if (photon.x() < nodes[node].p0.x())
            dmin += (photon.x() - nodes[node].p0.x()) * (photon.x() - nodes[node].p0.x());
        if (photon.y() > nodes[node].p1.y())
            dmin += (photon.y() - nodes[node].p1.y()) * (photon.y() - nodes[node].p1.y());
        if (photon.y() < nodes[node].p0.y())
            dmin += (photon.y() - nodes[node].p0.y()) * (photon.y() - nodes[node].p0.y());
        if (photon.z() > nodes[node].p1.z())
            dmin += (photon.z() - nodes[node].p1.z()) * (photon.z() - nodes[node].p1.z());
        if (photon.z() < nodes[node].p0.z())
            dmin += (photon.z() - nodes[node].p0.z()) * (photon.z() - nodes[node].p0.z());
        if (dmin > nodes[node].max_r_sqr)
            return;
        auto &pnt = hit_pnts[nodes[node].idx];
        if ((photon - pnt.p).len2() <= pnt.r_sqr && pnt.valid) {
            //auto dr = d - pnt.norm * (d.dot(pnt.norm) * 2);
            //double rho = pnt.brdf->rho_d + pnt.brdf->rho_s * pow(dr.dot(pnt.d), pnt.brdf->phong_s);
            pnt.delta_n++;
            //pnt.flux_i += pnt.weight * weight * clampVal(rho);
            pnt.flux_i += pnt.weight.mult(weight);
        }
        // recursively update children
        if (nodes[node].left != NULL_NODE)
            update(nodes[node].left, photon, weight, d);
        if (nodes[node].right != NULL_NODE)
            update(nodes[node].right, photon, weight, d);
        nodes[node].max_r_sqr = pnt.r_sqr;
        if (nodes[node].left != NULL_NODE)
            nodes[node].max_r_sqr = std::max(pnt.r_sqr, hit_pnts[nodes[nodes[node].left].idx].r_sqr);
        if (nodes[node].right != NULL_NODE)
            nodes[node].max_r_sqr = std::max(pnt.r_sqr, hit_pnts[nodes[nodes[node].right].idx].r_sqr);
    }

    uint32_t getRoot() const {
        return root;
    }
};
