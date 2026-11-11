package tensor

import "fmt"

func Fused0499SqrtMatmulFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_matmul_floor: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sqrt()
	r = r.Floor()
	return r, nil
}
func Fused0499SqrtMatmulFloorName() string { return "sqrt_matmul_floor" }
