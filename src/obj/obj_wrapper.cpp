//
// Created by throne on 19-5-12.
//
#include <tuple>
#include "obj_wrapper.h"
#include "OBJ_Loader.h"

std::tuple<std::vector<utils::Vector3>, std::vector<TriangleFace>, std::vector<utils::Vector3>> loadObject(const std::string fname) {
    objl::Loader Loader;
    std::vector<utils::Vector3> vertices;
    std::vector<utils::Vector3> norms;
    std::vector<TriangleFace> faces;
    if(Loader.LoadFile(fname)) {
        for (int i = 0; i < Loader.LoadedMeshes.size(); i++)
        {
            objl::Mesh curMesh = Loader.LoadedMeshes[i];
            for (int j = 0; j < curMesh.Vertices.size(); j++)
            {
                vertices.emplace_back(utils::Vector3(curMesh.Vertices[j].Position.X, curMesh.Vertices[j].Position.Y, curMesh.Vertices[j].Position.Z));
                norms.emplace_back(utils::Vector3(curMesh.Vertices[j].Normal.X, curMesh.Vertices[j].Normal.Y, curMesh.Vertices[j].Normal.Z));
                if((j + 1) % 3 == 0)
                {
                    TriangleFace triangle = {j - 2, j - 1, j, j - 2, j - 1, j, (int)faces.size()};
                    faces.emplace_back(triangle);
                }
            }
        }
    }
    else {
        fprintf(stderr, "Obj file failed to load!\n");
        exit(EXIT_FAILURE);
    }
    return {vertices, faces, norms};
}