package tensor

import "fmt"

func Fused0859CosMatmulFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_matmul_floor: nil operand")
	}
	r := x.MatMul(y)
	r = r.Cos()
	r = r.Floor()
	return r, nil
}
func Fused0859CosMatmulFloorName() string { return "cos_matmul_floor" }
