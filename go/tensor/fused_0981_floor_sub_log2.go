package tensor

import "fmt"

func Fused0981FloorSubLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_sub_log2: nil operand")
	}
	r := x.Sub(y)
	r = r.Floor()
	r = r.Log2()
	return r, nil
}
func Fused0981FloorSubLog2Name() string { return "floor_sub_log2" }
