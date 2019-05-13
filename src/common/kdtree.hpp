#pragma once

#include <vector>
#include <limits>
#include <memory.h>
#include "common.h"
#include "geometry.hpp"
#include "simple_intersect_algs.hpp"
#include "ray.hpp"

constexpr uint32_t MAX_FACES_PER_NODE = 10;
constexpr uint32_t NULL_NODE = std::numeric_limits<uint32_t>::max();

enum SplitAxis {
    X_AXIS = 0, Y_AXIS, Z_AXIS
};
enum {
    LEFT = 0, RIGHT, BOTTOM, TOP, BACK, FRONT
};

struct KDNode {
    uint32_t left, right;
    uint32_t ropes[6];
    std::vector<uint32_t> tri_list;
    utils::Vector3 p0, p1; // bounding boxes
    SplitAxis split_axis;
    double split_pos;
    uint32_t id;

    explicit KDNode(uint32_t id_ = 0) : id(id_) {
        left = right = NULL_NODE;
        for (size_t i = 0; i < 6; ++i)
            ropes[i] = NULL_NODE;
    }

    KDNode(KDNode &&other) : left(other.left), right(other.right), tri_list(std::move(other.tri_list)),
                             p0(other.p0), p1(other.p1), split_axis(other.split_axis), split_pos(other.split_pos),
                             id(other.id) {
        memcpy(this->ropes, other.ropes, sizeof(uint32_t) * 6);
    }

    KDNode(const KDNode &) = delete;

    KDNode &operator=(const KDNode &) = delete;

    inline bool isPointLeftOfSplittingPlane(const utils::Vector3 &p) const {
        switch (split_axis) {
            case X_AXIS:
                return p.x() < split_pos;
            case Y_AXIS:
                return p.y() < split_pos;
            case Z_AXIS:
                return p.z() < split_pos;
            default: // invalid case
                return false;
        }
    }

    inline uint32_t getNeighboringNode(utils::Vector3 &p) {
        if (fabs(p.x() - p0.x()) < EPSILON_KD)
            return ropes[LEFT];
        else if (fabs(p.x() - p1.x()) < EPSILON_KD)
            return ropes[RIGHT];
        else if (fabs(p.y() - p0.y()) < EPSILON_KD)
            return ropes[BOTTOM];
        else if (fabs(p.y() - p1.y()) < EPSILON_KD)
            return ropes[TOP];
        else if (fabs(p.z() - p0.z()) < EPSILON_KD)
            return ropes[BACK];
        else if (fabs(p.z() - p1.z()) < EPSILON_KD)
            return ropes[FRONT];
        else
            return NULL_NODE;
    }

    inline bool isLeaf() const {
        return left == NULL_NODE && right == NULL_NODE;
    }
};

class KDNode_Mem_Allocator {
private:
    std::vector<KDNode> node_list;
public:
    explicit KDNode_Mem_Allocator() = default;

    uint32_t acquire() {
        node_list.emplace_back(std::move(KDNode(node_list.size())));
        return node_list.size() - 1;
    }

    KDNode &operator[](uint32_t pos) { return node_list[pos]; }

    const KDNode &operator[](uint32_t pos) const { return node_list[pos]; }
};

class KDTree {
private:
    std::vector<utils::Vector3> vertices;
    std::vector<TriangleFace> faces;
    std::vector<utils::Vector3> norms;
    KDNode_Mem_Allocator mem;
    uint32_t root;
    uint32_t num_level, num_leaves;

    std::pair<utils::Vector3, utils::Vector3> getTightAABB(const std::vector<utils::Vector3> &vertices_) const {
        auto p1 = utils::Vector3(-INF, -INF, -INF), p0 = utils::Vector3(INF, INF, INF);
        for (const auto &x : vertices_) {
            p0 = min(p0, x);
            p1 = max(p1, x);
        }
        return {p0, p1};
    }

    std::pair<utils::Vector3, utils::Vector3> getTightAABB(const std::vector<uint32_t> &tri_list) const {
        auto p1 = utils::Vector3(-INF, -INF, -INF), p0 = utils::Vector3(INF, INF, INF);
        for (const auto &v : tri_list) {
            const auto &x = faces[v];
            p0 = min(p0, vertices[x.v1]);
            p0 = min(p0, vertices[x.v2]);
            p0 = min(p0, vertices[x.v3]);
            p1 = max(p1, vertices[x.v1]);
            p1 = max(p1, vertices[x.v2]);
            p1 = max(p1, vertices[x.v3]);
        }
        return {p0, p1};
    }

    inline double getMinTriFaceVal(uint32_t tri_idx, SplitAxis axis) {
        TriangleFace &tri = faces[tri_idx];
        auto v1 = vertices[tri.v1];
        auto v2 = vertices[tri.v2];
        auto v3 = vertices[tri.v3];
        if (axis == X_AXIS) {
            return std::min(v1.x(), std::min(v2.x(), v3.x()));
        } else if (axis == Y_AXIS) {
            return std::min(v1.y(), std::min(v2.y(), v3.y()));
        } else {
            return std::min(v1.z(), std::min(v2.z(), v3.z()));
        }
    }

    inline double getMaxTriFaceVal(uint32_t tri_idx, SplitAxis axis) {
        TriangleFace &tri = faces[tri_idx];
        auto v1 = vertices[tri.v1];
        auto v2 = vertices[tri.v2];
        auto v3 = vertices[tri.v3];
        if (axis == X_AXIS) {
            return std::max(v1.x(), std::max(v2.x(), v3.x()));
        } else if (axis == Y_AXIS) {
            return std::max(v1.y(), std::max(v2.y(), v3.y()));
        } else {
            return std::max(v1.z(), std::max(v2.z(), v3.z()));
        }
    }

    inline double getMidPointOfTriFace(uint32_t tri_idx, SplitAxis axis) {
        TriangleFace &tri = faces[tri_idx];
        auto v1 = vertices[tri.v1];
        auto v2 = vertices[tri.v2];
        auto v3 = vertices[tri.v3];
        if (axis == X_AXIS) {
            return (v1.x() + v2.x() + v3.x()) / 3;
        } else if (axis == Y_AXIS) {
            return (v1.y() + v2.y() + v3.y()) / 3;
        } else {
            return (v1.z() + v2.z() + v3.z()) / 3;
        }
    }

    uint32_t buildTree(std::vector<uint32_t> &&tri_list, int depth) {
        uint32_t node = mem.acquire();
        mem[node].tri_list = std::move(tri_list);
        auto aabb = getTightAABB(mem[node].tri_list);
        mem[node].p0 = aabb.first, mem[node].p1 = aabb.second;

        auto size = mem[node].tri_list.size();

        if (size < MAX_FACES_PER_NODE) {
            if (depth > num_level)
                num_level = depth;
            ++num_leaves;
            return node;
        }
        // divide along the longest side
        auto diff = aabb.second - aabb.first;
        SplitAxis longest = (diff.x() > diff.y() && diff.x() > diff.z()) ? X_AXIS : ((diff.y() > diff.z()) ? Y_AXIS
                                                                                                           : Z_AXIS);
        double mid = 0;
        for (const auto &x : mem[node].tri_list) {
            mid += getMidPointOfTriFace(x, longest);
        }
        mid /= size; // average mid points of triangular faces
        mem[node].split_axis = longest;
        mem[node].split_pos = mid;

        double min_face_val, max_face_val;

        std::vector<uint32_t> left_tris, right_tris;
        for (const auto &x : mem[node].tri_list) {
            if (getMidPointOfTriFace(x, longest) < mid)
                left_tris.push_back(x);
            else
                right_tris.push_back(x);
        }

        int matches = 0;
        /*for (const auto &x: right_tris) {
            for (const auto &y: left_tris) {
                if (vertices[faces[x].v1] == vertices[faces[y].v1] && vertices[faces[x].v2] == vertices[faces[y].v2]
                    && vertices[faces[x].v3] == vertices[faces[y].v3])
                    matches++;
            }
        }*/

        if (static_cast<double>(matches) / left_tris.size() < 0.5 &&
            static_cast<double>(matches) / right_tris.size() < 0.5) {
            auto v1 = buildTree(std::move(left_tris), depth + 1);
            mem[node].left = v1;
            auto v2 = buildTree(std::move(right_tris), depth + 1);
            mem[node].right = v2;
        }

        return node;
    }

    bool intersect(uint32_t node, const Ray &r, double &t, utils::Vector3 &norm) const {
        const KDNode &curr = mem[node];
        auto t_ = intersectAABB(r, curr.p0, curr.p1);
        if (t_ < INF) {
            if (curr.isLeaf()) {
                bool intersection_ok = false;
                for (const auto &x: curr.tri_list) {
                    const TriangleFace &tri = faces[x];
                    auto v1 = vertices[tri.v1];
                    auto v2 = vertices[tri.v2];
                    auto v3 = vertices[tri.v3];
                    auto n1 = norms[tri.n1];
                    auto n2 = norms[tri.n2];
                    auto n3 = norms[tri.n3];
                    auto res = intersectTrianglularFace(r, v1, v2, v3, n1, n2, n3);
                    if (res.first < INF) {
                        intersection_ok = true;
                        if (res.first < t)
                            t = res.first, norm = res.second;
                    }
                }
                return intersection_ok;
            } else {
                bool l_intersect = false, r_intersect = false;
                if (curr.left != NULL_NODE)
                    l_intersect = intersect(curr.left, r, t, norm);
                if (curr.right != NULL_NODE)
                    r_intersect = intersect(curr.right, r, t, norm);
                return l_intersect || r_intersect;
            }
        }
        return false;
    }

public:
    std::pair<utils::Vector3, utils::Vector3> getAABB() const {
        return {mem[root].p0 - EPSILON, mem[root].p1 + EPSILON};
    }

    KDTree(const KDTree &) = delete;

    KDTree &operator=(const KDTree &) = delete;

    KDTree(const std::vector<utils::Vector3> &vectices_, const std::vector<TriangleFace> &faces_,
           const std::vector<utils::Vector3> &norms_) : vertices(vectices_), faces(faces_), norms(norms_) {
        num_level = num_leaves = 0;
        std::vector<uint32_t> tri_list;
        tri_list.reserve(faces_.size());
        for (uint32_t i = 0; i < faces_.size(); ++i)
            tri_list.emplace_back(i);
        root = buildTree(std::move(tri_list), 1);

        printf("KD-Tree construction finished with %lu vertices and %lu faces\n", vectices_.size(), faces_.size());
    }

    std::pair<double, utils::Vector3> intersect(const Ray &r) const {
        double t = INF;
        utils::Vector3 norm;
        bool hit = intersect(root, r, t, norm);
        if (hit) {
            return {t, norm};
        } else
            return {INF, utils::Vector3()};
    }
};
