package tensor

import "fmt"

func Fused0319TanhMatmulFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_matmul_floor: nil operand")
	}
	r := x.MatMul(y)
	r = r.Tanh()
	r = r.Floor()
	return r, nil
}
func Fused0319TanhMatmulFloorName() string { return "tanh_matmul_floor" }
