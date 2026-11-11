package tensor

import "fmt"

func Fused0942FloorMulCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_mul_cos: nil operand")
	}
	r := x.Mul(y)
	r = r.Floor()
	r = r.Cos()
	return r, nil
}
func Fused0942FloorMulCosName() string { return "floor_mul_cos" }
