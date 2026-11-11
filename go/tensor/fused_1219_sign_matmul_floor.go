package tensor

import "fmt"

func Fused1219SignMatmulFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_matmul_floor: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sign()
	r = r.Floor()
	return r, nil
}
func Fused1219SignMatmulFloorName() string { return "sign_matmul_floor" }
