"""
Multi-Object Selection
======================


Example demonstrating multi object selection using mouse events.

Hovering the mouse over a cube will highlight it with a bounding box.
Clicking on a cube will select it. Double-clicking a cube will select
all the items from that group (because the group has a double-click
event handler). Holding shift will add to the selection.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

from functools import partial
from random import randint, random
import numpy as np
import wgpu
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
import gemmi
from pygfx.renderers.wgpu import (
    Binding,
    BaseShader,
    RenderMask,
    register_wgpu_render_function,
)
from pygfx.utils import unpack_bitfield

import pylinalg as la
import gemmi



# Custom object, material, and matching render function


class Triangle(gfx.WorldObject):
    mol=None
    mapping = []
    coords = []
    def setMol(self,mol):
        self.mol = mol
        self.coords = []
        for model in self.mol:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        self.coords.append(atom.pos.tolist())
                        self.mapping.append([atom,residue,chain,model])
        self.coords = np.array(coords).astype(np.float32)
        # center coords
        self.coords -= coords.mean(axis=0)

class TriangleMaterial(gfx.Material):
    uniform_type = dict(
        gfx.Material.uniform_type,
        color="4xf4",
    )

    def __init__(self, *, color="white", **kwargs):
        super().__init__(**kwargs)
        self.color = color

    @property
    def color(self):
        """The uniform color of the triangle."""
        return gfx.Color(self.uniform_buffer.data["color"])

    @color.setter
    def color(self, color):
        color = gfx.Color(color)
        self.uniform_buffer.data["color"] = color
        self.uniform_buffer.update_range(0, 99999)
        self._store.color_is_transparent = color.a < 1

    @property
    def color_is_transparent(self):
        """Whether the color is (semi) transparent (i.e. not fully opaque)."""
        # Note the use of the the _store to make this attribute trackable,
        # so that when it changes, the shader is updated automatically.
        return self._store.color_is_transparent

    def _wgpu_get_pick_info(self, pick_value):
        # This should match with the shader
        values = unpack_bitfield(pick_value, wobject_id=20, index=26, x=9, y=9)
        return {
            "vertex_index": values["index"],
            "point_coord": (values["x"] - 256.0, values["y"] - 256.0),
        }


@register_wgpu_render_function(Triangle, TriangleMaterial)
class TriangleShader(BaseShader):
    type = "render"

    def get_bindings(self, wobject, shared):
        geometry = wobject.geometry

        # This is how we set templating variables (dict-like access on the shader).
        # Look for "{{scale}}" in the WGSL code below.
        self["scale"] = 1.0

        # Three uniforms and one storage buffer with positions
        bindings = {
            0: Binding("u_stdinfo", "buffer/uniform", shared.uniform_buffer),
            1: Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer),
            2: Binding("u_material", "buffer/uniform", wobject.material.uniform_buffer),
            3: Binding(
                "s_positions", "buffer/read_only_storage", geometry.positions, "VERTEX"
            ),
        }
        self.define_bindings(0, bindings)
        return {
            0: bindings,
        }

    def get_pipeline_info(self, wobject, shared):
        # We draw triangles, no culling
        return {
            "primitive_topology": wgpu.PrimitiveTopology.triangle_list,
            "cull_mode": wgpu.CullMode.none,
        }

    def get_render_info(self, wobject, shared):
        material = wobject.material
        geometry = wobject.geometry

        # Determine number of vertices
        n = 3 * geometry.positions.nitems
        n_instances = 1
        # Define in what passes this object is drawn.
        # Using RenderMask.all is a good default. The rest is optimization.
        render_mask = wobject.render_mask
        if not render_mask:  # i.e. set to auto
            render_mask = RenderMask.all
            if material.is_transparent:
                render_mask = RenderMask.transparent
            elif material.color_is_transparent:
                render_mask = RenderMask.transparent
            else:
                render_mask = RenderMask.opaque

        return {
            "indices": (n, n_instances),
            "render_mask": render_mask,
        }

    def get_code(self):
        return """
        {$ include 'pygfx.std.wgsl' $}

        @vertex
        fn vs_main(@builtin(vertex_index) index: u32) -> Varyings {

            let vertex_index = i32(index) / 3;
            let sub_index = i32(index) % 3;

            // Transform object positition into NDC coords
            let model_pos = load_s_positions(vertex_index);  // vec3
            let world_pos = u_wobject.world_transform * vec4<f32>(model_pos, 1.0);
            let ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos;

            // List of relative positions, in logical pixels
            // based on the given radius
            let offset = u_stdinfo.projection_transform * vec4<f32>(1.0, 1.0, 0.0, 0.0);  //radius
            let triBase = 3.464;
            let triHeigth = 3.0;
            let triBaseHalf = triBase * 0.5;
            let triOffset = vec2<f32>(triBaseHalf, 1.0);
            var uvs = array<vec2<f32>, 3>(
                vec2<f32>(0, 0) - triOffset,
                vec2<f32>(triBaseHalf, triHeigth) - triOffset,
                vec2<f32>(triBase,0) - triOffset
            );
            var positions = array<vec2<f32>, 3>(
                uvs[0] * offset.xy,
                uvs[1] * offset.xy,
                uvs[2] * offset.xy,
            );
            // Get position for *this* corner
            let screen_factor = u_stdinfo.logical_size.xy / 2.0;
            let screen_pos_ndc = ndc_pos.xy + {{scale}} * positions[sub_index];
            // / screen_factor;

            // Set the output
            var varyings: Varyings;
            varyings.texcoord  = vec2<f32>(uvs[sub_index]);
            varyings.position  = vec4<f32>(screen_pos_ndc, ndc_pos.zw);
            varyings.world_pos = vec3<f32>(ndc_to_world_pos(vec4<f32>(screen_pos_ndc, ndc_pos.zw)));
            // varyings.world_pos  = vec4<f32>(1.0, 1.0, 1.0, 1.0);
            // Picking
            varyings.pick_idx = u32(vertex_index);
            return varyings;
        }

        struct OutputStruct {
            @location(0) color: vec4<f32>,
            @builtin(frag_depth) depth: f32
        }
        @fragment
        fn fs_main(varyings: Varyings) -> FragmentOutput  {
            // var out: OutputStruct;
            let lensqr = dot(varyings.texcoord.xy, varyings.texcoord.xy);
            if (lensqr > 1) {
                discard;
            }
            // Find normal
            let normal = normalize(vec3<f32>(varyings.texcoord.xy, sqrt(1.0 - lensqr)));
            // Find depth
            let clipZW = varyings.position.z * u_stdinfo.projection_transform[2].zw + u_stdinfo.projection_transform[3].zw;
            let eyeDepth = (0.5 + 0.5 * clipZW.x / clipZW.y) + 1.0 * (1.0 - normal.z);
            // float eyeDepth = LinearEyeDepth(input.position.z) + 1.0 * (1.0 - normal.z);
            let a = u_material.color.a * u_material.opacity;
            var the_color = vec4<f32>(eyeDepth,eyeDepth,eyeDepth, a);
            let out_color = vec4<f32>(srgb2physical(the_color.rgb), the_color.a * u_material.opacity);
            apply_clipping_planes(varyings.world_pos);
            var out = get_fragment_output(varyings.position.z, out_color);
            // out.depth = eyeDepth; // = varyings.position.z;
            // let a = u_material.color.a * u_material.opacity;
            // out.color = vec4<f32>(eyeDepth,eyeDepth,eyeDepth, eyeDepth);
            $$ if write_pick
            // The wobject-id must be 20 bits. In total it must not exceed 64 bits.
            out.pick = (
                pick_pack(u32(u_wobject.id), 20) +
                pick_pack(varyings.pick_idx, 26) +
                pick_pack(u32(varyings.texcoord.x + 256.0), 9) +
                pick_pack(u32(varyings.texcoord.y + 256.0), 9)
            );
            $$ endif
            return out;
        }
        """


canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)

# camera = gfx.PerspectiveCamera(70, 16 / 9)
# camera.local.z = 400
# camera.show_pos(((0, 0, 0)))
camera = gfx.OrthographicCamera(10, 10)


controller = gfx.OrbitController(camera, register_events=renderer)

scene = gfx.Scene()
scene.add(gfx.AmbientLight(), camera.add(gfx.DirectionalLight()))

geometry = gfx.box_geometry(40, 40, 40)
default_material = gfx.MeshPhongMaterial(pick_write=True)
selected_material = gfx.MeshPhongMaterial(color="#FF0000", pick_write=True)
hover_material = gfx.MeshPhongMaterial(color="#FFAA00", pick_write=True)

outline = gfx.BoxHelper(thickness=3, color="#fa0")
scene.add(outline)

ambient = gfx.AmbientLight("#fff", 0.1)
scene.add(ambient)

light = gfx.DirectionalLight("#aaaaaa")
light.local.x = -50
light.local.y = 50
light.cast_shadow = True

light.shadow.camera.width = 100
light.shadow.camera.height = 100
light.shadow.camera.update_projection_matrix()

scene.add(light.add(gfx.DirectionalLightHelper(30, show_shadow_extent=True)))

def set_material(material, obj):
    if isinstance(obj, gfx.Mesh):
        obj.material = material


def select(event):
    # when this event handler is invoked on non-leaf nodes of the
    # scene graph, event.target will still point to the leaf node that
    # originally triggered the event, so we use event.current_target
    # to get a reference to the node that is currently handling
    # the event, which can be a Mesh, a Group or None (the event root)
    obj = event.current_target
    
    # prevent propagation so we handle this event only at one
    # level in the hierarchy
    event.stop_propagation()

    # clear selection
    if selected_objects and "Shift" not in event.modifiers:
        while selected_objects:
            ob = selected_objects.pop()
            ob.traverse(partial(set_material, default_material))

    # if the background was clicked, we're done
    if isinstance(obj, gfx.Renderer):
        return
    
    atid = vertex_index = event.pick_info["vertex_index"]   
    print("select", obj, obj.mol, atid)
    print(obj.mapping[atid])
 
    # set selection (group or mesh)
    selected_objects.append(obj)
    obj.traverse(partial(set_material, selected_material))


def hover(event):
    obj = event.target
    print("hovering", obj, event.type)
    if event.type == "pointer_enter":
        obj.add(outline)
        if "vertex_index" in event.pick_info:
            print("hover", event.pick_info["vertex_index"])    
            # New position in 3D space
            new_position = np.array(obj.coords[event.pick_info["vertex_index"]])
            # Set the position of the BoxHelper
            # or use self.set_transform_by_aabb(aabb, scale)
            # make aabb arround the given atom
            aabb = [new_position - np.array([-1,-1,-1]), new_position + np.array([-1,-1,-1])]
            outline.set_transform_by_aabb(aabb, scale=1.)
        else :
            outline.set_transform_by_object(obj, "local", scale=1.1)
        
    elif outline.parent:
        outline.parent.remove(outline)


def random_rotation():
    return la.quat_from_euler(
        ((random() - 0.5) / 100, (random() - 0.5) / 100, (random() - 0.5) / 100)
    )


def animate():
    # def random_rot(obj):
    #     if hasattr(obj, "random_rotation"):
    #        obj.local.rotation = la.quat_mul(obj.random_rotation, obj.local.rotation)

    # scene.traverse(random_rot)
    renderer.render(scene, camera)
    canvas.request_draw()


filename = "1crn.pdb"
st = gemmi.read_structure(filename)
# bu = gemmi.make_assembly(st.assemblies[0], st[0], gemmi.HowToNameCopiedChain.AddNumber)

coords = []
for model in st:
    for chain in model:
        for residue in chain:
            for atom in residue:
                coords.append(atom.pos.tolist())
coords = np.array(coords).astype(np.float32)
# center coords
coords -= coords.mean(axis=0)
print(len(coords))

t = Triangle(
    gfx.Geometry(positions=coords),
    TriangleMaterial(color="yellow", pick_write=True),
)
t.setMol(st)
# t.receive_shadow = True
# t.cast_shadow = True
t.add_event_handler(select, "click")
t.add_event_handler(hover, "pointer_enter", "pointer_leave")
scene.add(t)

selected_objects = []

if __name__ == "__main__":
    renderer.add_event_handler(select, "click")

    # Build up scene
    for _ in range(4):
        group = gfx.Group()
        # group.random_rotation = random_rotation()
        # select the group
        group.add_event_handler(select, "double_click")
        # scene.add(group)

        for _ in range(10):
            cube = gfx.Mesh(geometry, default_material)
            cube.receive_shadow = True
            cube.cast_shadow = True
            cube.local.position = (
                randint(-200, 200),
                randint(-200, 200),
                randint(-200, 200),
            )
            # cube.random_rotation = random_rotation()
            cube.add_event_handler(select, "click")
            cube.add_event_handler(hover, "pointer_enter", "pointer_leave")
            group.add(cube)

    canvas.request_draw(animate)
    run()
