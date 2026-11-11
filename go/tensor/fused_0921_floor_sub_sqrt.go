package tensor

import "fmt"

func Fused0921FloorSubSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_sub_sqrt: nil operand")
	}
	r := x.Sub(y)
	r = r.Floor()
	r = r.Sqrt()
	return r, nil
}
func Fused0921FloorSubSqrtName() string { return "floor_sub_sqrt" }
