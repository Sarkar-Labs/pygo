package tensor

import "fmt"

func Fused0408AbsDivFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_div_floor: nil operand")
	}
	r := x.Div(y)
	r = r.Abs()
	r = r.Floor()
	return r, nil
}
func Fused0408AbsDivFloorName() string { return "abs_div_floor" }
