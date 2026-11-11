package tensor

import "fmt"

func Fused1579Log2MatmulFloor(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_matmul_floor: nil operand")
	}
	r := x.MatMul(y)
	r = r.Log2()
	r = r.Floor()
	return r, nil
}
func Fused1579Log2MatmulFloorName() string { return "log2_matmul_floor" }
