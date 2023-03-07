import gmsh
import sys

if len(sys.argv) < 2:
    print("Usage: " + sys.argv[0] + " file")
    exit()

gmsh.initialize()

gmsh.open(sys.argv[1])

print('Model ' + gmsh.model.getCurrent() + ' (' + str(gmsh.model.getDimension()) + 'D)')

entities = gmsh.model.getEntities()

for e in entities:
    dim = e[0]
    tag = e[1]

    # Get the mesh nodes for the entity (dim, tag):
    nodeTags, nodeCoords, nodeParams = gmsh.model.mesh.getNodes(dim, tag)

    # Get the mesh elements for the entity (dim, tag):
    elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(dim, tag)

    # * Type and name of the entity:
    type = gmsh.model.getType(e[0], e[1])
    name = gmsh.model.getEntityName(e[0], e[1])
    if len(name): name += ' '
    print("Entity " + name + str(e) + " of type " + type)

    # * Number of mesh nodes and elements:
    numElem = sum(len(i) for i in elemTags)
    print(" - Mesh has " + str(len(nodeTags)) + " nodes and " + str(numElem) +
          " elements")

    # * Upward and downward adjacencies:
    up, down = gmsh.model.getAdjacencies(e[0], e[1])
    if len(up):
        print(" - Upward adjacencies: " + str(up))
    if len(down):
        print(" - Downward adjacencies: " + str(down))

        # * Does the entity belong to physical groups?
        physicalTags = gmsh.model.getPhysicalGroupsForEntity(dim, tag)
        if len(physicalTags):
            s = ''
            for p in physicalTags:
                n = gmsh.model.getPhysicalName(dim, p)
                if n: n += ' '
                s += n + '(' + str(dim) + ', ' + str(p) + ') '
            print(" - Physical groups: " + s)

        # * Is the entity a partition entity? If so, what is its parent entity?
        partitions = gmsh.model.getPartitions(e[0], e[1])
        if len(partitions):
            print(" - Partition tags: " + str(partitions) + " - parent entity " +
                  str(gmsh.model.getParent(e[0], e[1])))

        # * List all types of elements making up the mesh of the entity:
        for t in elemTypes:
            name, dim, order, numv, parv, _ = gmsh.model.mesh.getElementProperties(
                t)
            print(" - Element type: " + name + ", order " + str(order) + " (" +
                  str(numv) + " nodes in param coord: " + str(parv) + ")")

# We can use this to clear all the model data:
gmsh.clear()

gmsh.finalize()