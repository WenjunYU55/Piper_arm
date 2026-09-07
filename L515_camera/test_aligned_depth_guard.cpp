#include "aligned_depth_guard.h"
#include <cassert>
#include <librealsense2/rs.hpp>
#include <librealsense2/hpp/rs_internal.hpp>
#include <iostream>
#include <vector>
int main(){
 rs2::software_device dev; auto ds=dev.add_sensor("Depth");auto cs=dev.add_sensor("Color");
 rs2_intrinsics di={8,6,3.5f,2.5f,8,8,RS2_DISTORTION_NONE,{0,0,0,0,0}};
 auto ci=di;ci.fx=12;ci.fy=12;
 auto dp=ds.add_video_stream({RS2_STREAM_DEPTH,0,0,8,6,30,2,RS2_FORMAT_Z16,di});
 auto cp=cs.add_video_stream({RS2_STREAM_COLOR,0,1,8,6,30,3,RS2_FORMAT_RGB8,ci});
 ds.add_read_only_option(RS2_OPTION_DEPTH_UNITS,.001f);
 dp.register_extrinsics_to(cp,{{1,0,0,0,1,0,0,0,1},{0,0,0}});
 rs2::frame_queue dq(4),cq(4);ds.open(dp);cs.open(cp);ds.start(dq);cs.start(cq);
 std::vector<uint16_t> z(48,400);std::vector<uint8_t> rgb(144,100);
 ds.on_video_frame({z.data(),[](void*){},16,2,100,RS2_TIMESTAMP_DOMAIN_HARDWARE_CLOCK,1,dp});
 cs.on_video_frame({rgb.data(),[](void*){},24,3,100,RS2_TIMESTAMP_DOMAIN_HARDWARE_CLOCK,1,cp});
 auto depth=dq.wait_for_frame();auto color=cq.wait_for_frame();
 for(bool has_color:{true,false}){
  rs2::frame_queue fq(1);
  rs2::processing_block composite([&](rs2::frame f,rs2::frame_source& src){std::vector<rs2::frame> v{f};if(has_color)v.push_back(color);src.frame_ready(src.allocate_composite_frame(v));});
  composite.start(fq);composite.invoke(depth);auto fs=fq.wait_for_frame().as<rs2::frameset>();
  rs2::align align(RS2_STREAM_COLOR);auto out=align.process(fs);auto d=out.get_depth_frame();
  auto k=d.get_profile().as<rs2::video_stream_profile>().get_intrinsics();
  std::cout<<"has_color="<<has_color<<" output_fx="<<k.fx<<" original_depth_fx="<<di.fx<<" color_fx="<<ci.fx<<" guard_admits="<<realsense2_camera::alignedDepthMatchesColor(d, has_color ? color : rs2::frame())<<std::endl;
  if(k.fx!=(has_color?ci.fx:di.fx)) return 2;
  assert(realsense2_camera::alignedDepthMatchesColor(d, has_color ? color : rs2::frame()) == has_color);
  assert(!realsense2_camera::alignedDepthMatchesColor(depth, color));
  assert(!realsense2_camera::alignedDepthMatchesColor(rs2::frame(), color));
 }
 ds.stop();cs.stop();ds.close();cs.close();
}
