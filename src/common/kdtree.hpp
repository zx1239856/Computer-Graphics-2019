#pragma once

#include <vector>
#include <limits>
#include <memory.h>
#include "common.h"
#include "geometry.hpp"
#include "simple_intersect_algs.hpp"
#include "ray.hpp"

#define __NVCC__
#ifdef __NVCC__
//#include "../gpu/cuda_helpers.h"
#endif

constexpr uint32_t MAX_FACES_PER_NODE = 10;
constexpr uint32_t NULL_NODE = std::numeric_limits<uint32_t>::max();

enum SplitAxis {
    X_AXIS = 0, Y_AXIS, Z_AXIS
};
enum Side {
    LEFT = 0, RIGHT, BOTTOM, TOP, BACK, FRONT
};

#ifdef  __NVCC__
struct KDNode_GPU {
    uint32_t left, right;
    uint32_t ropes[6];
    uint32_t start_tri_idx;
    uint32_t num_tris;
    utils::Vector3 p0, p1;
    SplitAxis split_axis;
    double split_pos;
};

struct KDTree_GPU {
    /*KernelArray<utils::Vector3> vertices;
    KernelArray<TriangleFace> faces;
    KernelArray<utils::Vector3> norms;
    KernelArray<KDNode_GPU> nodes;*/
    uint32_t root;
};
#endif

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

    inline uint32_t getNeighboringNode(utils::Vector3 &p) const {
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

    size_t size() const { return node_list.size(); }

    KDNode &operator[](uint32_t pos) { return node_list[pos]; }

    const KDNode &operator[](uint32_t pos) const { return node_list[pos]; }

    const std::vector<KDNode> &getNodeList() const { return node_list; }
};

class KDTree {
private:
    std::vector<utils::Vector3> vertices;
    std::vector<TriangleFace> faces;
    std::vector<utils::Vector3> norms;
    KDNode_Mem_Allocator mem;
    uint32_t root;
    uint32_t num_level, num_leaves;
    size_t total_tris_in_node;

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

    uint32_t
    buildTree(std::vector<uint32_t> &&tri_list, const utils::Vector3 &p0, const utils::Vector3 &p1, int depth) {
        uint32_t node = mem.acquire();
        mem[node].tri_list = std::move(tri_list);
        //auto aabb = getTightAABB(mem[node].tri_list);
        // Note that AABBs of adjacent nodes should not overlap, otherwise the stackless ropes would be scrambled...
        mem[node].p0 = p0, mem[node].p1 = p1;

        auto size = mem[node].tri_list.size();

        total_tris_in_node += size;

        if (size < MAX_FACES_PER_NODE) {
            if (depth > num_level)
                num_level = depth;
            ++num_leaves;
            return node;
        }
        // divide along the longest side
        auto diff = p1 - p0;
        SplitAxis longest = (diff.x() > diff.y() && diff.x() > diff.z()) ? X_AXIS : ((diff.y() > diff.z()) ? Y_AXIS
                                                                                                           : Z_AXIS);
        double mid = 0;
        for (const auto &x : mem[node].tri_list) {
            mid += getMidPointOfTriFace(x, longest);
        }
        mid /= size; // average mid points of triangular faces
        mem[node].split_axis = longest;
        mem[node].split_pos = mid;


        std::vector<uint32_t> left_tris, right_tris;
        for (const auto &x : mem[node].tri_list) {
            if (getMidPointOfTriFace(x, longest) < mid)
                left_tris.push_back(x);
            else
                right_tris.push_back(x);
        }

        utils::Vector3 lp1 = p1, rp0 = p0;
        if (longest == X_AXIS)
            lp1.x() = rp0.x() = mid;
        else if (longest == Y_AXIS)
            lp1.y() = rp0.y() = mid;
        else
            lp1.z() = rp0.z() = mid;

        auto v1 = buildTree(std::move(left_tris), p0, lp1, depth + 1);
        mem[node].left = v1;
        auto v2 = buildTree(std::move(right_tris), rp0, p1, depth + 1);
        mem[node].right = v2;

        return node;
    }

    void optimizeRopes(uint32_t (*ropes)[6], uint32_t node) {
        const auto &p0 = mem[node].p0;
        const auto &p1 = mem[node].p1;
        for (int i = 0; i < 6; ++i) {
            uint32_t &rope_node = (*ropes)[i];
            if (rope_node == NULL_NODE)
                continue;
            while (!mem[rope_node].isLeaf()) {
                if (i == LEFT || i == RIGHT) {
                    if (mem[rope_node].split_axis == X_AXIS)
                        rope_node = (i == LEFT) ? mem[rope_node].right : mem[rope_node].left;
                    else if (mem[rope_node].split_axis == Y_AXIS) {
                        if (mem[rope_node].split_pos < p0.y() - EPSILON_KD)
                            rope_node = mem[rope_node].right;
                        else if (mem[rope_node].split_pos > p1.y() + EPSILON_KD)
                            rope_node = mem[rope_node].left;
                        else
                            break;
                    } else {
                        if (mem[rope_node].split_pos < p0.z() - EPSILON_KD)
                            rope_node = mem[rope_node].right;
                        else if (mem[rope_node].split_pos > p1.z() + EPSILON_KD)
                            rope_node = mem[rope_node].left;
                        else
                            break;
                    }
                } else if (i == FRONT || i == BACK) {
                    if (mem[rope_node].split_axis == Z_AXIS)
                        rope_node = (i == BACK) ? mem[rope_node].right : mem[rope_node].left;
                    else if (mem[rope_node].split_axis == X_AXIS) {
                        if (mem[rope_node].split_pos < p0.x() - EPSILON_KD)
                            rope_node = mem[rope_node].right;
                        else if (mem[rope_node].split_pos > p1.x() + EPSILON_KD)
                            rope_node = mem[rope_node].left;
                        else
                            break;
                    } else {
                        if (mem[rope_node].split_pos < p0.y() - EPSILON_KD)
                            rope_node = mem[rope_node].right;
                        else if (mem[rope_node].split_pos > p1.y() + EPSILON_KD)
                            rope_node = mem[rope_node].left;
                        else
                            break;
                    }
                } else {
                    if (mem[rope_node].split_axis == Y_AXIS)
                        rope_node = (i == BOTTOM) ? mem[rope_node].right : mem[rope_node].left;
                    else if (mem[rope_node].split_axis == Z_AXIS) {
                        if (mem[rope_node].split_pos < p0.z() - EPSILON_KD)
                            rope_node = mem[rope_node].right;
                        else if (mem[rope_node].split_pos > p1.z() + EPSILON_KD)
                            rope_node = mem[rope_node].left;
                        else
                            break;
                    } else {
                        if (mem[rope_node].split_pos < p0.x() - EPSILON_KD)
                            rope_node = mem[rope_node].right;
                        else if (mem[rope_node].split_pos > p1.x() + EPSILON_KD)
                            rope_node = mem[rope_node].left;
                        else
                            break;
                    }
                }
            }
        }
    }

    void buildRopeRecursive(uint32_t node, uint32_t (*ropes)[6], bool is_single_ray) {
        if (mem[node].isLeaf()) {
            for (int i = 0; i < 6; ++i)
                mem[node].ropes[i] = (*ropes)[i];
        } else {
            if (is_single_ray)
                optimizeRopes(ropes, node);
            Side sl, sr;
            if (mem[node].split_axis == X_AXIS)
                sl = LEFT, sr = RIGHT;
            else if (mem[node].split_axis == Y_AXIS)
                sl = BOTTOM, sr = TOP;
            else
                sl = BACK, sr = FRONT;
            uint32_t rs_left[6], rs_right[6];
            for (int i = 0; i < 6; ++i)
                rs_left[i] = rs_right[i] = (*ropes)[i];
            rs_left[sr] = mem[node].right;
            buildRopeRecursive(mem[node].left, &rs_left, is_single_ray);
            rs_right[sl] = mem[node].left;
            buildRopeRecursive(mem[node].right, &rs_right, is_single_ray);
        }
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

    bool singleRayStacklessIntersect(uint32_t node, const Ray &r, double &t_entry, double &t_exit,
                                     utils::Vector3 &norm) const {
        bool intersect_ok = false;
        double t_entry_prev = -INF;

        while (t_entry < t_exit && t_entry > t_entry_prev) {
            t_entry_prev = t_entry;

            auto hit = r.getVector(t_entry);
            // down traversal to a leaf
            while (!mem[node].isLeaf())
                node = mem[node].isPointLeftOfSplittingPlane(hit) ? mem[node].left : mem[node].right;
            for (const auto &x: mem[node].tri_list) {
                const TriangleFace &tri = faces[x];
                auto v1 = vertices[tri.v1];
                auto v2 = vertices[tri.v2];
                auto v3 = vertices[tri.v3];
                auto n1 = norms[tri.n1];
                auto n2 = norms[tri.n2];
                auto n3 = norms[tri.n3];
                auto res = intersectTrianglularFace(r, v1, v2, v3, n1, n2, n3);
                if (res.first < t_exit) {
                    intersect_ok = true;
                    t_exit = res.first;
                    norm = res.second;
                }
            }

            auto aabb_intersect = intersectAABBInOut(r, mem[node].p0, mem[node].p1);
            if (aabb_intersect.first)
                t_entry = aabb_intersect.third;
            else
                break;

            auto hit_exit = r.getVector(t_entry);
            node = mem[node].getNeighboringNode(hit_exit);

            if (node == NULL_NODE)
                break;
        }

        return intersect_ok;
    }

public:
    std::pair<utils::Vector3, utils::Vector3> getAABB() const {
        return {mem[root].p0 - EPSILON, mem[root].p1 + EPSILON};
    }

    KDTree(const KDTree &) = delete;

    KDTree &operator=(const KDTree &) = delete;

    KDTree(const std::vector<utils::Vector3> &vertices_, const std::vector<TriangleFace> &faces_,
           const std::vector<utils::Vector3> &norms_) : vertices(vertices_), faces(faces_), norms(norms_) {
        num_level = num_leaves = total_tris_in_node = 0;
        std::vector<uint32_t> tri_list;
        tri_list.reserve(faces_.size());
        for (uint32_t i = 0; i < faces_.size(); ++i)
            tri_list.emplace_back(i);
        auto aabb = getTightAABB(vertices_);
        root = buildTree(std::move(tri_list), aabb.first, aabb.second, 1);
        buildRopeStructure();
        printf("KD-Tree construction finished with %lu vertices and %lu faces,\nTotal nodes: %lu, leaf nodes: %u, depth: %u\n",
               vertices_.size(), faces_.size(), mem.size(), num_leaves, num_level);
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

    std::pair<double, utils::Vector3> singleRayStacklessIntersect(const Ray &r) const {
        utils::Vector3 norm;
        auto res = intersectAABBInOut(r, mem[root].p0, mem[root].p1);
        double t_near = res.second, t_far = res.third;
        if (res.first) {
            bool hit = singleRayStacklessIntersect(root, r, t_near, t_far, norm);
            if (hit)
                return {t_far, norm};
            else
                return {INF, utils::Vector3()};
        } else
            return {INF, utils::Vector3()};
    }

    void buildRopeStructure() {
        uint32_t ropes[6] = {NULL_NODE, NULL_NODE, NULL_NODE, NULL_NODE, NULL_NODE, NULL_NODE};
        buildRopeRecursive(root, &ropes, true);
    }

#ifdef __NVCC__

    KDNode_GPU toGPU() const {
        std::vector<KDNode_GPU> cpu_nodes;
        const auto &node_list = mem.getNodeList();
        std::vector<uint32_t> tri_indices;
        tri_indices.reserve(total_tris_in_node);
        for (const auto &x : node_list) {
            KDNode_GPU node;
            node.start_tri_idx = tri_indices.size();
            tri_indices.reserve(tri_indices.size() + x.tri_list.size());
            std::copy(x.tri_list.begin(), x.tri_list.end(), std::back_inserter(tri_indices));
            node.left = x.left, node.right = x.right, node.p0 = x.p0, node.p1 = x.p1, node.num_tris = x.tri_list.size(), node.split_axis = x.split_axis, node.split_pos = x.split_pos;
            for (int i = 0; i < 6; ++i) {
                node.ropes[i] = x.ropes[i];
            }
            cpu_nodes.emplace_back(std::move(node));
        }
        // TODO - copy mem to GPU
    }

#endif
};
