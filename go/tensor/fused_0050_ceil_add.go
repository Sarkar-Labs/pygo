package tensor

import "fmt"

func Fused0050CeilAdd(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_add: nil operand")
	}
	r := x.Add(y)
	r = r.Ceil()
	return r, nil
}
func Fused0050CeilAddName() string { return "ceil_add" }
