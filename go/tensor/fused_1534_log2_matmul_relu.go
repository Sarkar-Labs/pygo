package tensor

import "fmt"

func Fused1534Log2MatmulRelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_matmul_relu: nil operand")
	}
	r := x.MatMul(y)
	r = r.Log2()
	r = r.ReLU()
	return r, nil
}
func Fused1534Log2MatmulReluName() string { return "log2_matmul_relu" }
