<template>
    <div>
        <input type="checkbox" id="todo-item" :checked="isDone" />
        <label :for="id">{{ label }} & id is {{ id }}</label>
    </div>
</template>
<script>

export default {
    props: {
        label: { required: true, type: String },
        done: { default: false, type: Boolean },
        id: { required: true, type: String },
    },
    data() {
        return {
            isDone: this.done,
        }
    }
};
</script>
