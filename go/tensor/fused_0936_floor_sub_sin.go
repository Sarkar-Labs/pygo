package tensor

import "fmt"

func Fused0936FloorSubSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_sub_sin: nil operand")
	}
	r := x.Sub(y)
	r = r.Floor()
	r = r.Sin()
	return r, nil
}
func Fused0936FloorSubSinName() string { return "floor_sub_sin" }
