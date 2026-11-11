package tensor

import "fmt"

func Fused0987FloorMulLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_mul_log10: nil operand")
	}
	r := x.Mul(y)
	r = r.Floor()
	r = r.Log10()
	return r, nil
}
func Fused0987FloorMulLog10Name() string { return "floor_mul_log10" }
